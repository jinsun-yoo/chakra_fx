import os


from nanogpt_model import Block, GPTConfig
from model_profiler import ModelProfiler

import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import Placement, Shard, Partial, Replicate
import logging
#torch._logging.set_logs(dynamo=logging.DEBUG, bytecode=True)

from torch.distributed.device_mesh import init_device_mesh
from typing import Callable 
from apply_configuration import apply_configuration
# Usage: torchrun --nproc-per-node=<number of processes> transformer.py

num_iters = 10
batch_size = 1 
sequence_length = 256
dtype = torch.bfloat16
num_transformer_layers = 1 


class NanoGptProfiler(ModelProfiler):
    def __init__(self, 
                 fxgraph_handler: Callable[[torch.fx.GraphModule],None],
                 use_pytorch_ir: bool,
                 run_custom_backend_all_rank: bool,
                 dse_config_filepath: str,
                ):

        self.name = f"nanogpt"
        config = GPTConfig(n_transformer_layers=num_transformer_layers)
        tp_model = Block(config).to(dtype)

        if dse_config_filepath is not None:
            tp_model = apply_configuration(tp_model, dse_config_filepath)
        else:
            world_size = int(os.environ["WORLD_SIZE"])
            device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
            tp_model = tp_model.to("cuda")

            # Parallelization plan. Tensor parallel
            tp_model = parallelize_module(
                module=tp_model,
                device_mesh=device_mesh,
                parallelize_plan={
                    "attn.c_attn": ColwiseParallel(),
                    "attn.c_proj": RowwiseParallel(),
                    "mlp.c_fc": ColwiseParallel(),
                    "mlp.c_proj": RowwiseParallel(),
                },
            )
        
        sample_input = torch.rand(
            batch_size, sequence_length, config.n_embd, dtype=dtype, device="cuda"
        )

        super().__init__(
                fxgraph_handler,
                use_pytorch_ir,
                tp_model,
                sample_input,
                run_custom_backend_all_rank
                )

    def run_training_session(self):
        super().compile_model()
        output = self.model(self.sample_input)
        torch.cuda.synchronize()

        output.sum().backward()
        torch.cuda.synchronize()

