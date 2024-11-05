import os
from typing import Callable

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from src.chakra_fx.profilers.apply_configuration import apply_configuration

from .model_profiler import ModelProfiler

# Usage: torchrun --nproc-per-node=<number of processes> transformer.py


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(12_288, 49_152)
        self.w2 = torch.nn.Linear(49_152, 12_288)

    def forward(self, x):
        x_1 = self.w1(x)
        x_2 = self.w2(x_1)
        return x_2


class SimpleModelProfiler(ModelProfiler):
    def __init__(
        self,
        fxgraph_handler: Callable[[torch.fx.GraphModule], None],
        use_pytorch_ir: bool,
        run_custom_backend_all_rank: bool,
        dse_config_filepath: str,
    ):
        self.name = "simplemodel"
        model = SimpleModel()

        if dse_config_filepath is not None:
            parallel_model = apply_configuration(model, dse_config_filepath)
        else:
            world_size = int(os.environ["WORLD_SIZE"])
            device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
            model = model.to("cuda")

            # Parallelization plan. Tensor parallel
            parallel_model = parallelize_module(
                module=model,
                device_mesh=device_mesh,
                parallelize_plan={
                    "w1": ColwiseParallel(),
                    "w2": RowwiseParallel(),
                },
            )

        sample_input = torch.rand(1024, 12_288, dtype=torch.float, device="cuda")

        super().__init__(
            fxgraph_handler,
            use_pytorch_ir,
            parallel_model,
            sample_input,
            run_custom_backend_all_rank,
        )

    def run_training_session(self):
        super().compile_model()
        output = self.model(self.sample_input)
        torch.cuda.synchronize()

        output.sum().backward()
        torch.cuda.synchronize()

    def run_eager(self):
        num_range = os.environ['NUM_RANGE']
        if num_range != '':
            num_range = int(num_range)
        else:
            num_range = 1
        if os.environ['RANK'] == '0':
            print(f'num_range is {num_range}')
        output = self.model(self.sample_input)
        torch.cuda.synchronize()
        output.sum().backward()
        torch.cuda.synchronize()

    def collect_postexecution_graph(self, dirname: str, name: str):
        from torch.profiler import ExecutionTraceObserver, profile

        rank = os.environ["RANK"]

        def kineto_trace_handler(prof):
            prof.export_chrome_trace(f"{dirname}/{name}_kineto_rank{rank}.json")

        et = ExecutionTraceObserver()
        et.register_callback(f"{dirname}/{name}_pytorch_et_rank_{rank}.json")
        et.start()

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=10, active=1),
            on_trace_ready=kineto_trace_handler,
            with_flops=True,
        ) as prof:
            for epoch in range(20):
                if epoch == 19:
                    et.stop()
                if epoch == 10:
                    et.start()
                output = self.model(self.sample_input)
                torch.cuda.synchronize()
                output.sum().backward()
                torch.cuda.synchronize()
                prof.step()
        et.stop()
        et.unregister_callback()

    def run_nsys_workload(self):
        nb_iters = 20
        warmup_iters = 10
        for i in range(nb_iters):
            # start profiling after 10 warmup iterations
            if i == warmup_iters:
                torch.cuda.cudart().cudaProfilerStart()
            # push range for current iteration
            if i >= warmup_iters:
                torch.cuda.nvtx.range_push("iteration{}".format(i))

            # push range for forward
            if i >= warmup_iters:
                torch.cuda.nvtx.range_push("forward")
            output = self.model(self.sample_input)
            torch.cuda.synchronize()
            if i >= warmup_iters:
                torch.cuda.nvtx.range_pop()

            if i >= warmup_iters:
                torch.cuda.nvtx.range_push("backward")
            output.sum().backward()
            if i >= warmup_iters:
                torch.cuda.nvtx.range_pop()

            # pop iteration range
            if i >= warmup_iters:
                torch.cuda.nvtx.range_pop()

            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
