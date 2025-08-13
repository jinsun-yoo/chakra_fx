## Environment Setup
Our approach requires 1) Tricking PyTorch into thinking it's running on multiple GPUs, when it is actually running on only 1 GPU, and 2) disabling *actual* communication during parallelization before the model is even run/compiled (for example, Rank 0 sharding the weights across Ranks in TP as initial setup). This requires us to modify PyTorch code, which is... *fun*. There are largely two options:

### Docker based option
This option uses an existing Dockerfile. The container installs a specific version of PyTorch (`2.9.0.dev20250704+cu129`), and directly modifies the *installed python code* under `/usr`. The Dockerfile automatically applies the necessary patches. This is more fit if you simply want to try out this code.

> Note: As of today, we do not have to go into modifying C++ code. If we do have to modify C++ code, then... the only option is to build from source (refer to below)

Open the Dockerfile. Adjust the userid, username, groupid, groupname arguments as necessary. Then:
```bash
docker build -t chakra_fx:latest .
bash run_docker.sh
root@CONTAINER_HASH:/workspace# su ${YOUR_USERNAME_IN_THE_DOCKERFILE}
YOUR_USERNAME_IN_DOCKERIFLE@CONTAINER_HASH:/workspace$ cd chakra_fx/
YOUR_USERNAME_IN_DOCKERIFLE@CONTAINER_HASH:/workspace/chakra_fx$

# Check that the installed PyTorch has been modified correctly
# The second 'mesh_scatter' should have been commented out.
cat  /usr/local/lib/python3.12/dist-packages/torch/distributed/tensor/placement_types.py  | grep -nr 'mesh_scatter'
13:    mesh_scatter,
183:        #mesh_scatter(
```

### Code based option
This is more fit if you want to develop (modify the Pytorch code), or are in an environment where you cannot run Docker (Slurm)
Follow the build instructions in PyTorch's [README.md](https://github.com/pytorch/pytorch?tab=readme-ov-file#prerequisites) and [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#build-only-what-you-need)
(i.e. apart from cloning the specific branch/fork of the PyTorch repo, everything else is same)
> IF this is your first time, STRONGLY RECOMMEND doing this within a Docker container. Rever to the below "Combination of the above" for more.
```bash
# START of optional CCache for faster subsequent build
# Refer to [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#use-ccache)
sudo apt-get install ccache
ccache -M 25Gi
ccache -F 0
# END of optional CCache code.

git clone --recurse-submodules --branch chakra_fx https://github.com/jinsun-yoo/pytorch
cd pytorch
pip install cmake ninja
pip install -r requirements.txt
python -m pip install --no-build-isolation -v -e .
DEBUG=1 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 \
python -m pip install --no-build-isolation -v -e .
```

### A combination of the above.
You can build from source within a docker container (by mounting a local clone of the Python repository, or cloning the Python repo from within the container) OR you can install a specific version of python locally with `pip` and apply the changes directly to the installed container. For me, I take the 'Docker container' + 'Mount local version of Python code' approach. To mount a local version of the python code, simply add a mount line to `run_docker.sh`.
```
...
	-v /nethome/jyoo332/chakra_fx:/workspace/chakra_fx \
	-v /nethome/jyoo332/pytorch:/workspace/pytorch \
...
```

## Running Examples
### Toy examples that doesn't generate Chakra graphs, but show you how you can extract FXGraph w/o multiple GPUs.
#### Simple AllGather on a tensor
```bash
cd minimal_repro/tp
torchrun --nproc-per-node=4 sample.py
```
#### Simple TP on a linear module
```bash
cd minimal_repro/tp
torchrun --nproc-per-node=4 sample2.py
```
#### FSDP across 2 layers of transformer model
```bash
cd minimal_repro/fsdp
torchrun --nproc-per-node=4 sample2.py
```
### If you're interested in debugging
```bash
export TORCH_LOGS=+dynamo,graph,graph_code,bytecode
torchrun ...
```

### Extracting Chakra traces
```bash
cd chakra_fx #top directory
# IF using within docker container, 
# MUST switch to user owning chakra_fx
su {USER_NAME} 
mkdir test_output
torchrun --nproc-per-node=4 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama
ls test_output
trace.0.et  trace.1.et  trace.2.et  trace.3.et
chakra_jsonizer --input_filename trace.0.et --output_filename trace.0.json
cat trace.0.json
```

### Extracting Chakra traces for different configs
```bash

torchrun --nproc-per-node=8 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama --dse_config_filepath configs/llama_FSDP_8.yml

torchrun --nproc-per-node=4 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw
 --model llama --dse_config_filepath configs/llama_FSDP_4.yml

torchrun --nproc-per-node=8 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama --dse_config_filepath configs/llama_TP_8.yml

```
