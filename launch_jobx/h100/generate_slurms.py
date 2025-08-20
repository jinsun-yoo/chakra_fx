slurm_template = """#!/bin/bash
#SBATCH -J %s
#SBATCH -N %d
#SBATCH --gres=%s
#SBATCH --mem=256000
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:15:00
#SBATCH -o logs/%s.%%j.out
#SBATCH -e logs/%s.%%j.err

set -euo pipefail
module load anaconda3
conda activate chakra_fx

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^lo,docker0

NPROC_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=$((12000 + SLURM_JOB_ID %% 20000))

echo "MASTER_ADDR=$MASTER_ADDR NNODES=$SLURM_NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
cd /home/hice1/cman8/code/chakra_fx

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  profile_fxgraph.py \
    --fxgraph_actions chakra \
    --exp_tag %s \
    --job eager_kineto \
    --model %s \
    --dse_config_filepath %s

export CUDA_VISIBLE_DEVICES=0
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  torchrun \
  --nnodes=1 \
  --nproc_per_node=%d \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  profile_fxgraph.py \
    --fxgraph_actions chakra \
    --exp_tag %s \
    --job fwbw \
    --model %s \
    --dse_config_filepath %s
    
"""


def generate_slurm(job_name, model, dse_config, num_gpus, num_nodes, gpu_type):
    slurm_script = slurm_template % (
        job_name,
        num_nodes,
        gpu_type,
        job_name,
        job_name,
        job_name,
        model,
        dse_config,
        num_gpus,
        job_name,
        model,
        dse_config,
    )
    return slurm_script


# fmt: off
configs = [
    # ("dp4_llama_small", "llama_small", "configs/llama_dp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("dp8_llama_small", "llama_small", "configs/llama_dp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    ("dp16_llama_small", "llama_small", "configs/llama_dp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("dp32_llama_small", "llama_small", "configs/llama_dp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("dp64_llama_small", "llama_small", "configs/llama_dp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # ("fsdp4_llama_small", "llama_small", "configs/llama_fsdp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("fsdp8_llama_small", "llama_small", "configs/llama_fsdp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    ("fsdp16_llama_small", "llama_small", "configs/llama_fsdp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("fsdp32_llama_small", "llama_small", "configs/llama_fsdp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("fsdp64_llama_small", "llama_small", "configs/llama_fsdp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # ("tp4_llama_small", "llama_small", "configs/llama_tp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("tp8_llama_small", "llama_small", "configs/llama_tp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    ("tp16_llama_small", "llama_small", "configs/llama_tp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("tp32_llama_small", "llama_small", "configs/llama_tp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("tp64_llama_small", "llama_small", "configs/llama_tp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    
    # # ("dp4_llama_tiny", "llama_tiny", "configs/llama_dp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("dp8_llama_tiny", "llama_tiny", "configs/llama_dp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("dp16_llama_tiny", "llama_tiny", "configs/llama_dp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("dp32_llama_tiny", "llama_tiny", "configs/llama_dp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("dp64_llama_tiny", "llama_tiny", "configs/llama_dp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # # ("fsdp4_llama_tiny", "llama_tiny", "configs/llama_fsdp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("fsdp8_llama_tiny", "llama_tiny", "configs/llama_fsdp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("fsdp16_llama_tiny", "llama_tiny", "configs/llama_fsdp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("fsdp32_llama_tiny", "llama_tiny", "configs/llama_fsdp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("fsdp64_llama_tiny", "llama_tiny", "configs/llama_fsdp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # # ("tp4_llama_tiny", "llama_tiny", "configs/llama_tp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("tp8_llama_tiny", "llama_tiny", "configs/llama_tp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("tp16_llama_tiny", "llama_tiny", "configs/llama_tp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("tp32_llama_tiny", "llama_tiny", "configs/llama_tp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("tp64_llama_tiny", "llama_tiny", "configs/llama_tp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),

    # # ("dp4_llama_3b", "llama_3b", "configs/llama_dp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("dp8_llama_3b", "llama_3b", "configs/llama_dp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("dp16_llama_3b", "llama_3b", "configs/llama_dp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("dp32_llama_3b", "llama_3b", "configs/llama_dp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("dp64_llama_3b", "llama_3b", "configs/llama_dp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # # ("fsdp4_llama_3b", "llama_3b", "configs/llama_fsdp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("fsdp8_llama_3b", "llama_3b", "configs/llama_fsdp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("fsdp16_llama_3b", "llama_3b", "configs/llama_fsdp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("fsdp32_llama_3b", "llama_3b", "configs/llama_fsdp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("fsdp64_llama_3b", "llama_3b", "configs/llama_fsdp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # # ("tp4_llama_3b", "llama_3b", "configs/llama_tp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("tp8_llama_3b", "llama_3b", "configs/llama_tp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("tp16_llama_3b", "llama_3b", "configs/llama_tp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("tp32_llama_3b", "llama_3b", "configs/llama_tp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("tp64_llama_3b", "llama_3b", "configs/llama_tp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    
    # # ("dp4_llama_1b", "llama_1b", "configs/llama_dp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("dp8_llama_1b", "llama_1b", "configs/llama_dp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("dp16_llama_1b", "llama_1b", "configs/llama_dp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("dp32_llama_1b", "llama_1b", "configs/llama_dp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("dp64_llama_1b", "llama_1b", "configs/llama_dp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # # ("fsdp4_llama_1b", "llama_1b", "configs/llama_fsdp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("fsdp8_llama_1b", "llama_1b", "configs/llama_fsdp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("fsdp16_llama_1b", "llama_1b", "configs/llama_fsdp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("fsdp32_llama_1b", "llama_1b", "configs/llama_fsdp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("fsdp64_llama_1b", "llama_1b", "configs/llama_fsdp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
    # # ("tp4_llama_1b", "llama_1b", "configs/llama_tp/llama_FSDP_4.yml", 4, 1, "gpu:H100:8",),
    # ("tp8_llama_1b", "llama_1b", "configs/llama_tp/llama_FSDP_8.yml", 8, 1, "gpu:H100:8",),
    # ("tp16_llama_1b", "llama_1b", "configs/llama_tp/llama_FSDP_16.yml", 16, 2, "gpu:H100:8",),
    # ("tp32_llama_1b", "llama_1b", "configs/llama_tp/llama_FSDP_32.yml", 32, 4, "gpu:H100:8",),
    # ("tp64_llama_1b", "llama_1b", "configs/llama_tp/llama_FSDP_64.yml", 64, 8, "gpu:H100:8",),
]
# fmt: on

for dp in configs:
    job_name = dp[0]
    with open(f"job_{job_name}.slurm", "w") as f:
        slurm_script = generate_slurm(*dp)
        f.write(slurm_script)
        print(f"Generated {job_name}.slurm")
