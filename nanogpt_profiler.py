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

from torch.distributed._tensor.device_mesh import init_device_mesh
from typing import Callable 
# Usage: torchrun --nproc-per-node=<number of processes> transformer.py

num_iters = 10
batch_size = 16 
sequence_length = 256
#batch_size = 64
#sequence_length = 2048
dtype = torch.bfloat16
world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
rank = device_mesh.get_rank()


class NanoGptProfiler(ModelProfiler):
    def __init__(self, 
                 fxgraph_handler: Callable[[torch.fx.GraphModule],None],
                 use_pytorch: bool
                ):
        self.name = "nanoGPT"

        config = GPTConfig()
        tp_model = Block(config).to(dtype).to("cuda")
        tp_model = torch.compile(tp_model)


        # Parallelization plan. Tensor parallel
        tp_model = parallelize_module(
            module=tp_model,
            device_mesh=device_mesh,
            parallelize_plan={
                "attn.c_attn_key": ColwiseParallel(),
                "attn.c_attn_query": ColwiseParallel(),
                "attn.c_attn_value": ColwiseParallel(),
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
        output.sum().backward()

        output = self.model(self.sample_input)
        torch.cuda.synchronize()

        output.sum().backward()

