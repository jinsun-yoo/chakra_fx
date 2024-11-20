import subprocess

for parallelization in ["FSDP8", "TP8",  "TP4FSDP2"]:#,"TP2FSDP4"]:
    command = [
        "nsys", "profile", "-w", "true", "-t", "cuda,nvtx,osrt,cudnn,cublas,mpi,openmp", 
        "--capture-range", "cudaProfilerApi", "--cudabacktrace", "true", "-x", "true", 
        "-o", f"/my_workspace/data/20241114_Llama3_debug_{parallelization}/report", 
        "torchrun", "--nproc-per-node=8", "profile_fxgraph.py", 
        "--custom_backend_all_rank", "False", "--dse_config_filepath", f"./configs/simple_{parallelization}.yml", 
        "--actions", "chakra", "--exp_tag", f"/my_workspace/data/20241114_Llama3_debug_{parallelization}", 
        "--job", "nsys", "--model", "llama"
    ]
    subprocess.run(command)

