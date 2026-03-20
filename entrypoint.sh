#!/bin/bash
set -euo pipefail

# PyTorchJob injects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE.
# GPUS_PER_NODE is set in the manifest as an env var.
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NUM_NODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_NODE="${MASTER_ADDR:-localhost}"
MASTER_NODE_PORT="${MASTER_PORT:-29500}"

echo "============================================"
echo "  Multi-Node SFT Trainer"
echo "============================================"
echo "  Node Rank:      ${NODE_RANK}"
echo "  Num Nodes:      ${NUM_NODES}"
echo "  GPUs/Node:      ${GPUS_PER_NODE}"
echo "  Master:         ${MASTER_NODE}:${MASTER_NODE_PORT}"
echo "============================================"

exec torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_NODE}" \
    --master_port="${MASTER_NODE_PORT}" \
    /app/train.py "$@"
