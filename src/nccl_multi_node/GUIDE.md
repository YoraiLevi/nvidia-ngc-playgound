# Multi-Node NCCL with Rootless Podman

Run a GPU-to-GPU communication test across **two DGX Spark nodes** entirely
without root, using Podman containers.

---

## What this proves

Both Spark GPUs (one per node) talk to each other through NCCL over the
ConnectX-7 (CX-7) RDMA network, and the whole thing runs inside rootless
Podman containers -- no `sudo`, no Docker daemon, no `docker` group.

## Prerequisites

| Requirement | How to check |
|---|---|
| Two DGX Sparks connected by a QSFP cable | `ibdev2netdev` shows at least one `(Up)` interface on each node |
| Passwordless SSH from Node 1 to Node 2 | `ssh <node2-hostname> hostname` works without a password prompt |
| NVIDIA driver + CUDA toolkit on both nodes | `nvidia-smi` and `/usr/local/cuda/bin/nvcc --version` |
| Podman installed on both nodes | `podman --version` |
| CDI configured on Node 1 **or** manual device passthrough (see below) | `podman run --rm --device nvidia.com/gpu=all ... nvidia-smi` |

> **CDI note:** If `/etc/cdi/nvidia.yaml` exists the simpler
> `--device nvidia.com/gpu=all` flag works. If it does not (and you have no
> root access to create it) the included `run_nccl_podman.sh` script passes
> every device file manually -- no CDI needed.

---

## Steps

There are three things to do: build NCCL, compile the test, pull the
container image.  Then you run one command on each node.

### 1. Build NCCL from source (both nodes)

Run this on **Node 1**, then repeat on **Node 2** (via SSH or a second
terminal).

```bash
# Clone NCCL (Blackwell-compatible version)
git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl/

# Build (sm_121 = Blackwell / GB10)
cd ~/nccl/
make -j$(nproc) src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
```

On Node 2 the CUDA toolkit may not be on `$PATH`; prefix the `make` with:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

This takes about 5 minutes per node.  Once done you will have
`~/nccl/build/lib/libnccl.so.2`.

### 2. Compile the test program (both nodes)

```bash
nvcc -o ~/nccl_podman_test \
     src/nccl_multi_node/nccl_podman_test.cu \
     -I ~/nccl/build/include \
     -L ~/nccl/build/lib -lnccl -lcudart \
     -gencode=arch=compute_121,code=sm_121
```

Copy the binary to Node 2:

```bash
scp ~/nccl_podman_test <node2>:~/nccl_podman_test
```

(Or just compile on Node 2 the same way after copying the `.cu` file.)

### 3. Pull the container image (both nodes)

```bash
podman pull docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04
```

On Node 2 via SSH:

```bash
ssh <node2> 'podman pull docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04'
```

The image is ~420 MB.

### 4. Find your CX-7 IP addresses

On each node:

```bash
# See which CX-7 ports are up (only look at enp1... interfaces)
ibdev2netdev

# Get the IP of the first up interface (example: enp1s0f0np0)
ip -4 addr show enp1s0f0np0
```

Write down the IP for each node.  Example:

| Node | IP |
|---|---|
| Node 1 (head) | 192.168.200.10 |
| Node 2 (worker) | 192.168.200.11 |

### 5. Run the test

#### Option A -- single command with `spark-podman-run` (recommended)

From the repo root on Node 1:

```bash
src/nccl_multi_node/spark-podman-run \
  -v ~/nccl/build/lib:/nccl-lib:ro \
  -v ~/nccl_podman_test:/app/nccl_test:ro \
  -e 'LD_LIBRARY_PATH=/nccl-lib:/cuda-lib:/host-lib' \
  -e 'NCCL_DEBUG=INFO' \
  docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04 -- \
    bash -c '/app/nccl_test $RANK 2 $MASTER_ADDR'
```

This auto-detects Node 2 from `~/.ssh/config`, discovers the CX-7
interface and IPs, rsyncs the workspace, and launches both containers in
parallel with color-coded output (blue = node1, yellow = node2).

Add `--dry-run` to see the generated `podman run` commands without
executing them.

#### Option B -- manual two-terminal approach

You need two terminals (or one terminal + an SSH session).

**Terminal 1 -- Node 1 (rank 0, starts first):**

```bash
src/nccl_multi_node/run_nccl_podman.sh 0 2 <NODE1_IP>
```

It will print "Listening ... for rank 1" and wait.

**Terminal 2 -- Node 2 (rank 1):**

```bash
# Either SSH in and run locally:
ssh <node2>
~/run_nccl_podman.sh 1 2 <NODE1_IP>

# Or from Node 1 via SSH:
ssh <node2> '~/run_nccl_podman.sh 1 2 <NODE1_IP>'
```

> Replace `<NODE1_IP>` with the CX-7 IP of Node 1 (e.g. `192.168.200.10`).
> Both ranks point to **Node 1's** IP -- rank 0 listens on it, rank 1
> connects to it.

### 6. Read the output

Both terminals should end with:

```
[Rank N] AllReduce result: PASS
[Rank N]   Send value: X.0, Result: 3.0, Expected: 3.0
[Rank N]   Errors: 0 / 1048576
...
[Rank N] AllGather result: PASS
...
[Rank N] *** SUCCESS: Multi-node NCCL communication verified in rootless podman! ***
```

Key lines to look for in the NCCL `INFO` log:

| Line | What it means |
|---|---|
| `Using network IB` | NCCL is using RDMA (RoCE) over the CX-7 NIC |
| `nRanks 2 nNodes 2` | Two separate Spark nodes are participating |
| `Connected all rings` | All communication channels are established |
| `via NET/IB/0` | Data flows over the InfiniBand/RoCE transport |

---

## How it works (short version)

```
┌─ Node 1 ────────────────────────┐   CX-7 RDMA    ┌─ Node 2 ────────────────────────┐
│  podman run (rootless)          │◄──────────────►│  podman run (rootless)          │
│    ├ --device /dev/nvidia0      │   (RoCE)        │    ├ --device /dev/nvidia0      │
│    ├ --device /dev/infiniband/* │                  │    ├ --device /dev/infiniband/* │
│    ├ --network=host             │                  │    ├ --network=host             │
│    └ nccl_podman_test (rank 0)  │                  │    └ nccl_podman_test (rank 1)  │
│       └ NVIDIA GB10 GPU         │                  │       └ NVIDIA GB10 GPU         │
└─────────────────────────────────┘                  └─────────────────────────────────┘
```

1. Rank 0 generates an `ncclUniqueId` and sends it to rank 1 over a TCP
   socket.
2. Both ranks call `ncclCommInitRank()` -- NCCL discovers the CX-7 RDMA NIC
   and sets up ring communication channels.
3. `ncclAllReduce` and `ncclAllGather` move data between the two GPUs.
4. The test verifies every element of the result buffer matches the expected
   value.

The containers get GPU access through raw `--device` flags (no CDI
required), host library mounts (`-v`), and `--network=host` for direct
CX-7 access.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `bind: Address already in use` | A previous run left a socket open.  Wait 30 s or change `BOOTSTRAP_PORT` in the `.cu` file and recompile. |
| `NCCL error ... no CUDA-capable device` | Make sure `/dev/nvidia0` and `/dev/nvidia-uvm` are passed with `--device`. |
| Rank 1 hangs on "Connecting to rank 0" | Check that both nodes can reach each other on the CX-7 IPs (`ping <ip>`). |
| `ibdev2netdev` shows all interfaces `(Down)` | The QSFP cable is not plugged in, or the link has not come up yet. |
| `libnccl.so.2: cannot open shared object` | `~/nccl/build/lib/` does not exist on this node.  Re-run step 1. |

---

## Files

```
src/nccl_multi_node/
├── nccl_podman_test.cu    ← test source (CUDA + NCCL, no MPI)
├── run_nccl_podman.sh     ← podman launch wrapper (auto-detects NIC)
├── spark-podman-run       ← single-command orchestrator for both nodes
└── GUIDE.md               ← this file
```
