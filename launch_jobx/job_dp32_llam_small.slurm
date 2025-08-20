#!/bin/bash
#SBATCH -A gts-tkrishna3
#SBATCH -J dp32_llam_small
#SBATCH -N 8
#SBATCH -q embers
#SBATCH --gres=gpu:RTX_6000:4
#SBATCH --mem=256000
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
module load anaconda3
conda activate chakra_fx

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^lo,docker0

NPROC_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR NNODES=$SLURM_NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
cd /storage/home/hcoda1/1/cman8/code/chakra_fx

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
  --nnodes=8 \
  --nproc_per_node=$NPROC_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  profile_fxgraph.py \
    --fxgraph_actions chakra \
    --exp_tag dp32_llamasmall \
    --job eager_kineto \
    --model llama_tiny \
    --dse_config_filepath configs/llama_dp/llama_FSDP_32.yml
