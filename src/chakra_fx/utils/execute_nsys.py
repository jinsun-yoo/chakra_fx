import os
import subprocess

for parallelization in [
    # "FSDP16",
    # "TP16",
    # "TP2FSDP8",
    # "TP4FSDP4",
    "TP8FSDP2"
]:
    if not os.path.exists(f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}"):
        os.mkdir(f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}")
    """
    command = [
        "nsys", "profile", "-w", "true", "-t", "cuda,nvtx,osrt,cudnn,cublas,mpi,openmp",
        "--capture-range", "cudaProfilerApi", "--cudabacktrace", "true", "-x", "true",
        "-o", f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}/report",
        "torchrun", "--nproc-per-node=8", "profile_fxgraph.py",
        "--custom_backend_all_rank", "False", "--dse_config_filepath", f"./configs/simple_{parallelization}.yml",
        "--actions", "chakra", "--exp_tag", f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}",
        "--job", "nsys", "--model", "llama"
    ]
    """
    """
    command = [
        "nsys", "profile", "-w", "true", "-t", "cuda,nvtx,osrt,cudnn,cublas,mpi,openmp",
        "--capture-range", "cudaProfilerApi", "--cudabacktrace", "true", "-x", "true",
        "-o", f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}/report",
        "torchrun", "--nproc-per-node=8",
        "--nnodes=2", "--node-rank=0", "--rdzv-id=456", "--rdzv-backend=c10d",
        "--rdzv-endpoint=10.184.206.136:29603", "--rdzv-conf=is_host=true",
        "profile_fxgraph.py",
        "--custom_backend_all_rank", "False", "--dse_config_filepath", f"./configs/simple_{parallelization}.yml",
        "--actions", "chakra", "--exp_tag", f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}",
        "--job", "nsys", "--model", "llama"
    ]
    """
    command = [
        "torchrun",
        "--nproc-per-node=8",
        "--nnodes=2",
        "--node-rank=0",
        "--rdzv-id=456",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=10.184.206.136:29603",
        "--rdzv-conf=is_host=true",
        "profile_fxgraph.py",
        "--custom_backend_all_rank",
        "False",
        "--dse_config_filepath",
        f"./configs/simple_{parallelization}.yml",
        "--actions",
        "chakra",
        "--exp_tag",
        f"/my_workspace/data/20241131_Llama3_chakrafxmodel_{parallelization}",
        "--job",
        "sample",
        "--model",
        "llama",
    ]
    """
    command = [
        "python", "torchrun_astrasim_launcher.py", "--config_filepath", f"./configs/simple_{parallelization}.yml",
        "--output_dir", f"/my_workspace/data/20241120_Llama3_chakrafxmodel_{parallelization}"
    ]
    """
    print(" ".join(command))
    subprocess.run(command)
