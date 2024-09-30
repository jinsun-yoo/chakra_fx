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
import logging
#torch._logging.set_logs(dynamo=logging.DEBUG, bytecode=True)

from torch.distributed._tensor.device_mesh import init_device_mesh
from typing import Callable 
# Usage: torchrun --nproc-per-node=<number of processes> transformer.py

num_iters = 10
batch_size = 1 
sequence_length = 256
#batch_size = 64
#sequence_length = 2048
dtype = torch.bfloat16
world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
rank = device_mesh.get_rank()
num_transformer_layers = 1 


class NanoGptProfiler(ModelProfiler):
    def __init__(self, 
                 fxgraph_handler: Callable[[torch.fx.GraphModule],None],
                 use_pytorch: bool
                ):
        self.name = f"nanoGPT_nightly_{num_transformer_layers}_layers"

        config = GPTConfig(n_transformer_layers=num_transformer_layers)
        tp_model = Block(config).to(dtype).cuda(dist.get_rank())

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
                use_pytorch,
                tp_model,
                sample_input
                )

    def run_training_session(self):
        super().compile_model()
        output = self.model(self.sample_input)
        torch.cuda.synchronize()

        output.sum().backward()
        torch.cuda.synchronize()

