# chakra_fx


# Get the chakra trace from a llama model, and store it in the directory '/my_workspace'

```
torchrun        --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:29500 --nnodes=2 --nproc-per-node=8  --log-dir /my_workspace         -r 3    /workspace/chakra_fx/profile_fxgraph.py         --custom_backend_all_rank True      --dse_config_filepath /workspace/chakra_fx/configs/simple_FSDP16.yml     --actions chakra        --exp_tag /my_workspace --job sample      --model llama
```

# Get the nsys trace from a llama model, and store it in the directory '/my_workspace'
```
nsys profile -w true    -t cuda,nvtx,osrt,cudnn,cublas,mpi,openmp --capture-range cudaProfilerApi     --cudabacktrace true    -x true         -o /my_workspace/report$SLURM_NODEID         torchrun        --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:29500 --nnodes=2 --nproc-per-node=8  --log-dir /my_workspace         -r 3    /workspace/chakra_fx/profile_fxgraph.py         --custom_backend_all_rank True      --dse_config_filepath /workspace/chakra_fx/configs/simple_TP2FSDP8.yml     --actions chakra        --exp_tag /my_workspace --job nsys      --model llama
```

# Get the kineto, pytorch trace from a llama model, and store it in the directory '/my_workspace'
```
torchrun        --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:29500 --nnodes=2 --nproc-per-node=8  --log-dir /my_workspace         -r 3    /workspace/chakra_fx/profile_fxgraph.py         --custom_backend_all_rank True      --dse_config_filepath /workspace/chakra_fx/configs/simple_FSDP16.yml    --exp_tag /my_workspace --job postexec      --model llama
```

# Running the above using slurm batch
```
sbatch -p {CLUSTER_NAME} slurm.sh
```

# Running inside docker container
```
docker run  -it --rm -p 8888:8888 --gpus all --network=host --uts=host --ipc=host --ulimit stack=67108864 --ulimit memlock=-1 --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-e TERM=screen-256color \
-v /mnt/nvdl/usr/${USER}/:/my_workspace/ \
${1:-'github.com'} /bin/bash
```

