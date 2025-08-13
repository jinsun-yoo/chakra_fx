#!/bin/bash
#SBATCH --job-name=profile-chakra
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=champollion

set -x

module load openmpi
conda activate python312

# Define NUM_RANKS variable
NUM_RANKS=4


echo "Hostname: $(hostname)"
echo "SLURM_NODENAME: $SLURMD_NODENAME"
# Use srun instead of mpirun for better SLURM integration
#--mpi=pmix_v3 \
srun \
    --output=%j_output_rank_%t.log \
    --error=%j_error_rank_%t.log \
    --ntasks=${NUM_RANKS} \
    bash -c "source ~/.bashrc; conda init; conda activate python312; which python; torchrun \
    --nnodes ${NUM_RANKS} \
    --nproc-per-node=1 \
    --rdzv-id=456 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$(hostname):29500 \
    collect_chakra.py"
# --logging "${PROJECT_DIR}/logger_config.toml" \