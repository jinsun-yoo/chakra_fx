#!/bin/bash
#SBATCH --job-name=slurm
#SBATCH --output=slurm%j.out
#SBATCH --error=slurm%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=8
#SBATCH --switches=1@10
set -x
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

echo "nodes $nodes"

# Print the job ID and the hostname of the node where the job is running
echo "Job ID: $SLURM_JOBID"
echo "Hostname: $SLURM_JOB_NODELIST"
nvidia-smi
# Run a simple bash command
echo "Hello, world!"
export COMPILE="True"
srun --nodes=2 --gpus=8 --container-image=$CONTAINER_IMAGE_PATH \
	--container-mounts=/labhome/jinsuny/20241208_FSDP16:/my_workspace \
		bash -c "torchrun        --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:29500 --nnodes=2 --nproc-per-node=8  --log-dir /my_workspace         -r 3    /workspace/chakra_fx/profile_fxgraph.py         --custom_backend_all_rank True      --dse_config_filepath /workspace/chakra_fx/configs/simple_FSDP16.yml     --actions chakra        --exp_tag /my_workspace --job sample      --model llama "
		
