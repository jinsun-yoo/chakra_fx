import re
import argparse
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument('--config_filepath', type=str, default="", required=False, help="Path to the configuration file. Mutually exclusive with 'config_*' options below.")
    parser.add_argument('--output_dir', type=str, default="", required=False, help="Path to the output directory where Chakra ET will be stored")

    # Config Args
    parser.add_argument('--config_parallelization', type=str, default="", help="Parallelization strategy. Oneof: 'fsdp', 'tp', 'tp,fsdp'. Mutually exclusive with 'config_filepath' option.")
    parser.add_argument('--config_tp', type=int, default=1, help="Degree of Tensor Parallelism. 'config_tp' x 'config_fsdp' is the total number of GPU.")
    parser.add_argument('--config_fsdp', type=int, default=1, help="Degree of FSDP Parallelism. 'config_tp' x 'config_fsdp' is the total number of GPU")

    args = parser.parse_args()
    return args 

def generate_config_file(args) -> str:
    """Based on the command line arguments, create the configuration file to pass to 'profile_fxgraph'."""
    import yaml
    config_yaml_dict = {}
    tp_struct = {
            "layers": {
                "attn.c_attn": "Colwise",
                "attn.c_proj": "Rowwise",
                "mlp.c_fc": "Colwise",
                "mlp.c_proj": "Rowwise"
            }
        }
    parallelization_list = args.config_parallelization.split(",")
    dimensions_list = []
    for parallelization_strategy in parallelization_list:
        if parallelization_strategy == "tp":
            config_yaml_dict['tp'] = tp_struct
            dimensions_list.append(args.config_tp)
        if parallelization_strategy == "fsdp":
            dimensions_list.append(args.config_fsdp)
            config_yaml_dict['fsdp'] = {}
    config_yaml_dict['overall'] = {
            'dimensions': dimensions_list,
            'parallelization': parallelization_list
            }

    config_filename = f'{output_dir}/config.yml'
    with open(config_filename, 'w') as f:
        yaml.dump(config_yaml_dict, f, default_flow_style=True)
    return config_filename


if __name__ == "__main__":
    args = parse_args()

    config_filepath = args.config_filepath
    output_dir = args.output_dir

    if output_dir == "":
        from datetime import datetime as dt, timezone
        now_utc = dt.now(timezone.utc)
        output_dir = formatted_dt = now_utc.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(f'./{output_dir}'):
        os.mkdir(f'./{output_dir}')

    if config_filepath=="":
        config_filepath=generate_config_file(args)
    
    # TODO: Dynamic values in multi-node run
    total_num_gpus = 8

    # TODO: Multi-node run
    # Run the torchrun command to execute 'profile_fxgraph.py' script across multiple GPUs.
    # profile_fxgraph.py will perform PyTorch Model -> FX Graph -> Chakra Graph conversion. 
    result = subprocess.run(['/usr/local/bin/torchrun',
                             '--nproc_per_node', f'{total_num_gpus}',
                             '/workspace/chakra_fx/profile_fxgraph.py', 
                             '--custom_backend_all_rank', 'True',
                             '--dse_config_filepath', config_filepath,
                             '--actions', 'chakra',
                             '--exp_tag', output_dir], stdout=subprocess.PIPE)

    # TODO: Dynamic ASTRA-sim network layer input for multi-node run.
    # TODO: Dynamic ASTRA-sim system layer input depending on DSE search space.
    # Run the astra-sim command with the Chakra Graph generated above as workload input. 
    astrasim_dir = "/workspace/astra-sim"
    binary = f"{astrasim_dir}/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    workload = f"{output_dir}/nanogpt"
    system = f"{astrasim_dir}/inputs/system/Ring.json"
    network = f"{astrasim_dir}/inputs/network/analytical/Switch_8.yml"
    remote_mem = f"{astrasim_dir}/inputs/remote_memory/analytical/no_memory_expansion.json"
    result = subprocess.run([binary,
                             f'--workload-configuration={workload}',
                             f'--system-configuration={system}',
                             f'--network-configuration={network}',
                             f'--remote-memory-configuration={remote_mem}'], capture_output=True)

    print(result.stderr)
    print(result.stdout)

