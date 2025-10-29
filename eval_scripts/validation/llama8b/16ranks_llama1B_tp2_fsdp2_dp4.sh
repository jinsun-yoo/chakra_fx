export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^lo,docker0

NPROC_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR NNODES=$SLURM_NNODES NPROC_PER_NODE=$NPROC_PER_NODE"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    ./profile_fxgraph.py \
    --exp_tag llama8B_tp2_fsdp2_dp4_8ranks \
    --job postexec_chakra \
    --model llama \
    --dse_config_filepath configs/llama_16/llama_TP2_FSDP2_DP4.yml \
    --llama_config 8B
