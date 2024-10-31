import yaml
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ParallelStyle,
    RowwiseParallel,
    ColwiseParallel
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._composable.fsdp import fully_shard


def axis_to_style(axis_name : str) -> ParallelStyle:
    if axis_name == "Rowwise":
        return RowwiseParallel()
    if axis_name == "Colwise":
        return ColwiseParallel()

def apply_tensor_parallel(model, tp_data, device_mesh):
    parallelize_plan = {}
    for layer_fqn, axis in tp_data['layers'].items():
        parallelize_plan[layer_fqn] = axis_to_style(axis)
    model = parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )
    return model

def apply_fsdp(model, fsdp_data, device_mesh):
    model = FSDP(model, device_mesh=device_mesh, use_orig_params=True)
    return model

# This code is motivated from the following code in torchtitan.
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def apply_configuration(model, dse_config_filepath):
    # Load the YAML file
    with open(dse_config_filepath, 'r') as file:
        data = yaml.safe_load(file)

    if data is None:
        # Default, no prarallelization.
        return model
    
    # Get the dimensions from the YAML file
    dimensions = data['overall']['dimensions']
    dim_parallelizations = data['overall']['parallelization']

    # Initialize the device mesh
    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=tuple(dimensions), mesh_dim_names=tuple(dim_parallelizations))
    model = model.to("cuda")

    # Apply parallelization strategy
    # TODO: Since we now know that FSDP2 does not work with 2D parallelism, this code has to be fixed to use SimpleFSDP.
    # The Torchtitan repo code already uses SimpleFSDP.
    for _, parallelization in enumerate(dim_parallelizations):
        if parallelization == "tp":
            parallelized_model = apply_tensor_parallel(model, data['tp'], device_mesh[parallelization])
        elif parallelization == "fsdp":
            parallelized_model = apply_fsdp(model, data['fsdp'], device_mesh[parallelization])
        else:
            raise(f"Error, parallelization strategy {parallelization} not found!")

    # Return the parallelized model
    return parallelized_model
