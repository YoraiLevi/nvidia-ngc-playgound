/*
 * Standalone multi-node NCCL test for rootless Podman containers.
 *
 * Two processes (one per DGX Spark) exchange an ncclUniqueId over a plain
 * TCP socket, then run AllReduce and AllGather across both GPUs.
 * No MPI dependency -- only NCCL and CUDA.
 *
 * Build (on each node):
 *   nvcc -o nccl_podman_test nccl_podman_test.cu \
 *        -I ~/nccl/build/include -L ~/nccl/build/lib -lnccl -lcudart \
 *        -gencode=arch=compute_121,code=sm_121
 *
 * Run (see run_nccl_podman.sh or the guide for the full podman command):
 *   Node 1:  ./nccl_podman_test 0 2 <node1_cx7_ip>
 *   Node 2:  ./nccl_podman_test 1 2 <node1_cx7_ip>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define NCCLCHECK(cmd) do {                                  \
  ncclResult_t r = cmd;                                      \
  if (r != ncclSuccess) {                                    \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",               \
        __FILE__, __LINE__, ncclGetErrorString(r));          \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)

#define CUDACHECK(cmd) do {                                  \
  cudaError_t e = cmd;                                       \
  if (e != cudaSuccess) {                                    \
    fprintf(stderr, "CUDA error %s:%d '%s'\n",               \
        __FILE__, __LINE__, cudaGetErrorString(e));          \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)

#define BOOTSTRAP_PORT 18517

/* Rank 0: generate the NCCL unique ID and send it to rank 1 over TCP. */
static void exchange_id_rank0(ncclUniqueId *id, const char *listen_ip) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(BOOTSTRAP_PORT);
    inet_pton(AF_INET, listen_ip, &addr.sin_addr);
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); exit(1);
    }
    listen(server_fd, 1);
    printf("[Rank 0] Listening on %s:%d for rank 1...\n", listen_ip, BOOTSTRAP_PORT);
    fflush(stdout);
    int client_fd = accept(server_fd, NULL, NULL);
    send(client_fd, id, sizeof(*id), 0);
    close(client_fd);
    close(server_fd);
    printf("[Rank 0] Sent NCCL unique ID to rank 1\n");
}

/* Rank 1: connect to rank 0 and receive the NCCL unique ID. */
static void exchange_id_rank1(ncclUniqueId *id, const char *rank0_ip) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(BOOTSTRAP_PORT);
    inet_pton(AF_INET, rank0_ip, &addr.sin_addr);
    printf("[Rank 1] Connecting to rank 0 at %s:%d...\n", rank0_ip, BOOTSTRAP_PORT);
    fflush(stdout);
    int retries = 0;
    while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        if (++retries > 100) { perror("connect"); exit(1); }
        usleep(200000);
    }
    ssize_t n = recv(sock, id, sizeof(*id), MSG_WAITALL);
    close(sock);
    if (n != (ssize_t)sizeof(*id)) { fprintf(stderr, "Failed to receive ID\n"); exit(1); }
    printf("[Rank 1] Received NCCL unique ID from rank 0\n");
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <rank> <world_size> <ip>\n", argv[0]);
        fprintf(stderr, "  rank 0: ip = address to listen on\n");
        fprintf(stderr, "  rank 1: ip = rank 0's address\n");
        return 1;
    }

    int rank = atoi(argv[1]);
    int nRanks = atoi(argv[2]);
    const char *ip = argv[3];

    printf("========================================\n");
    printf("[Rank %d] NCCL Multi-Node Podman Test\n", rank);
    printf("[Rank %d] World size: %d\n", rank, nRanks);
    printf("========================================\n");
    fflush(stdout);

    CUDACHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[Rank %d] GPU: %s\n", rank, prop.name);

    ncclUniqueId id;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
        exchange_id_rank0(&id, ip);
    } else {
        exchange_id_rank1(&id, ip);
    }

    printf("[Rank %d] Initializing NCCL communicator...\n", rank);
    fflush(stdout);
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));
    printf("[Rank %d] NCCL communicator initialized successfully!\n", rank);

    /* --- Test 1: AllReduce ------------------------------------------------ */
    int N = 1024 * 1024;
    float *sendbuf, *recvbuf;
    CUDACHECK(cudaMalloc(&sendbuf, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuf, N * sizeof(float)));

    float fill_val = (float)(rank + 1);
    float *host_buf = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) host_buf[i] = fill_val;
    CUDACHECK(cudaMemcpy(sendbuf, host_buf, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));

    printf("[Rank %d] Running AllReduce (SUM) on %d elements (%.2f MB)...\n",
           rank, N, (float)(N * sizeof(float)) / (1024*1024));
    fflush(stdout);

    NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, N, ncclFloat, ncclSum, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));

    CUDACHECK(cudaMemcpy(host_buf, recvbuf, N * sizeof(float), cudaMemcpyDeviceToHost));
    float expected = (float)(nRanks * (nRanks + 1) / 2);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (host_buf[i] != expected) {
            errors++;
            if (errors < 5)
                printf("[Rank %d] Mismatch at %d: got %f expected %f\n",
                       rank, i, host_buf[i], expected);
        }
    }

    printf("\n========================================\n");
    printf("[Rank %d] AllReduce result: %s\n", rank, errors == 0 ? "PASS" : "FAIL");
    printf("[Rank %d]   Send value: %.1f, Result: %.1f, Expected: %.1f\n",
           rank, fill_val, host_buf[0], expected);
    printf("[Rank %d]   Errors: %d / %d\n", rank, errors, N);
    printf("========================================\n");

    /* --- Test 2: AllGather ------------------------------------------------ */
    int M = 4 * 1024 * 1024;
    float *ag_send, *ag_recv;
    CUDACHECK(cudaMalloc(&ag_send, M * sizeof(float)));
    CUDACHECK(cudaMalloc(&ag_recv, M * nRanks * sizeof(float)));

    float *ag_host = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) ag_host[i] = (float)(rank * 100 + i % 100);
    CUDACHECK(cudaMemcpy(ag_send, ag_host, M * sizeof(float), cudaMemcpyHostToDevice));

    printf("[Rank %d] Running AllGather on %d elements per rank (%.2f MB total)...\n",
           rank, M, (float)(M * nRanks * sizeof(float)) / (1024*1024));
    fflush(stdout);

    NCCLCHECK(ncclAllGather(ag_send, ag_recv, M, ncclFloat, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));

    float *ag_result = (float*)malloc(M * nRanks * sizeof(float));
    CUDACHECK(cudaMemcpy(ag_result, ag_recv, M * nRanks * sizeof(float), cudaMemcpyDeviceToHost));

    int ag_errors = 0;
    for (int r = 0; r < nRanks; r++) {
        float exp_first = (float)(r * 100);
        if (ag_result[r * M] != exp_first) ag_errors++;
    }

    printf("[Rank %d] AllGather result: %s\n", rank, ag_errors == 0 ? "PASS" : "FAIL");
    printf("========================================\n");
    printf("[Rank %d] NCCL VERSION: %d.%d.%d\n", rank, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
    if (errors == 0 && ag_errors == 0) {
        printf("[Rank %d] *** SUCCESS: Multi-node NCCL communication "
               "verified in rootless podman! ***\n", rank);
    } else {
        printf("[Rank %d] *** FAILURE: NCCL communication errors "
               "detected ***\n", rank);
    }
    printf("========================================\n");
    fflush(stdout);

    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaFree(sendbuf));
    CUDACHECK(cudaFree(recvbuf));
    CUDACHECK(cudaFree(ag_send));
    CUDACHECK(cudaFree(ag_recv));
    CUDACHECK(cudaStreamDestroy(s));
    free(host_buf);
    free(ag_host);
    free(ag_result);

    return (errors > 0 || ag_errors > 0) ? 1 : 0;
}
