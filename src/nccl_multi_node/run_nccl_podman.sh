#!/bin/bash
# Run the NCCL multi-node test inside a rootless Podman container.
#
# Usage:
#   ./run_nccl_podman.sh <rank> <world_size> <ip> [interface]
#
#   rank 0  — ip is this node's CX-7 address (it listens for rank 1)
#   rank 1  — ip is rank 0's CX-7 address   (it connects to rank 0)
#   interface (optional) — CX-7 NIC name, e.g. enp1s0f0np0.
#             Auto-detected from ibdev2netdev if omitted.
set -euo pipefail

RANK=${1:?Usage: $0 <rank> <world_size> <ip> [interface]}
WORLD_SIZE=${2:?}
IP=${3:?}

# --- auto-detect CX-7 interface (first "enp1..." that is Up) -------------
if [[ -n "${4:-}" ]]; then
    IF_NAME="$4"
else
    IF_NAME=$(ibdev2netdev 2>/dev/null \
        | grep -E '^rocep1' \
        | grep '(Up)' \
        | head -1 \
        | awk '{print $5}')
    if [[ -z "$IF_NAME" ]]; then
        echo "ERROR: could not auto-detect a CX-7 interface. Pass it as \$4." >&2
        exit 1
    fi
fi

IB_HCA=$(ibdev2netdev 2>/dev/null \
    | grep "$IF_NAME" \
    | awk '{print $1}')

IMAGE="docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04"

echo "=== NCCL Podman Test ==="
echo "  Rank:       $RANK / $WORLD_SIZE"
echo "  IP:         $IP"
echo "  Interface:  $IF_NAME  (IB HCA: ${IB_HCA:-none})"
echo "  User:       $(whoami) (UID $(id -u)) — no root"
echo "  Runtime:    $(podman --version)"
echo ""

PODMAN_ARGS=(
  --rm
  --network=host
  --ipc=host
  # GPU devices
  --device /dev/nvidia0
  --device /dev/nvidiactl
  --device /dev/nvidia-uvm
  --device /dev/nvidia-uvm-tools
  # RDMA / InfiniBand devices
  --device /dev/infiniband/rdma_cm
  --device /dev/infiniband/uverbs0
  --device /dev/infiniband/uverbs1
  --device /dev/infiniband/uverbs2
  --device /dev/infiniband/uverbs3
  # Mount host libraries into the container
  -v /usr/local/cuda/targets/sbsa-linux/lib:/cuda-lib:ro
  -v /usr/lib/aarch64-linux-gnu:/host-lib:ro
  -v "$HOME/nccl/build/lib:/nccl-lib:ro"
  -v "$HOME/nccl_podman_test:/app/nccl_test:ro"
  # Library search path
  -e LD_LIBRARY_PATH="/nccl-lib:/cuda-lib:/host-lib"
  # NCCL configuration
  -e NCCL_SOCKET_IFNAME="$IF_NAME"
  -e NCCL_DEBUG=INFO
)

[[ -n "${IB_HCA:-}" ]] && PODMAN_ARGS+=(-e NCCL_IB_HCA="$IB_HCA")

exec podman run "${PODMAN_ARGS[@]}" \
  "$IMAGE" \
  /app/nccl_test "$RANK" "$WORLD_SIZE" "$IP"
