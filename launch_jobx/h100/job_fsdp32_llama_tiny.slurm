#!/bin/bash
#SBATCH -J fsdp32_llama_tiny
#SBATCH -N 4
#SBATCH --gres=gpu:H100:8
#SBATCH --mem=256000
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:14:00
#SBATCH -o logs/fsdp32_llama_tiny.%j.out
#SBATCH -e logs/fsdp32_llama_tiny.%j.err

set -euo pipefail
module load anaconda3
conda activate chakra_fx

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^lo,docker0

NPROC_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))

echo "MASTER_ADDR=$MASTER_ADDR NNODES=$SLURM_NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
cd /home/hice1/cman8/code/chakra_fx

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1   torchrun   --nnodes=$SLURM_NNODES   --nproc_per_node=$NPROC_PER_NODE   --rdzv_backend=c10d   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT   --rdzv_id=$SLURM_JOB_ID   profile_fxgraph.py     --fxgraph_actions chakra     --exp_tag fsdp32_llama_tiny     --job eager_kineto     --model llama_tiny     --dse_config_filepath configs/llama_fsdp/llama_FSDP_32.yml
    
