import os
from typing import List

import torch.fx
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from src.chakra_fx.passes.custom_compiler import build_custom_backend_compiler


class ModelProfiler:
    def __init__(
        self,
        model,
        sample_input,
        sample_label,
        exp_tag: str,
        fxgraph_actions: List[str],
        run_custom_backend_all_rank: bool,
        use_pytorch_ir: bool = False,
    ):
        self.rank = int(os.environ.get("RANK", 0))
        self.size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # If true, work on PyTorch FX Graph, if false, work on aten FX Graph
        self.use_pytorch_ir = use_pytorch_ir
        # Index of subgraph at each graph break. Increments with each subgraph
        self.subgraph_idx = 0
        # This name will be used at, e.g. start of generated filenames
        if not hasattr(self, "name"):
            print("Name not provided. Use default name.")
            self.name = "modelProfiler"

        # Unless otherwise noted, run custom backend instead of the default inductor backend.
        # Would set run_custom_backend_all_rank to False for e.g. debugging purposes.
        self.run_custom_backend = run_custom_backend_all_rank or self.local_rank == 0

        # TODO: WHen parsing FXGraph, should not specify rank.
        # TODO: But when parsing chakra trace, SHOULD specify rank.
        self.fxgraph_handler = build_custom_backend_compiler(fxgraph_actions, exp_tag, self)
        self.model = model
        self.sample_input = sample_input
        self.sample_label = sample_label
        return

    def _custom_pytorch_compiler(self, gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        self.fxgraph_handler(gm, self)
        self.subgraph_idx += 1
        return gm.forward

    def _custom_aten_compiler(self, gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        self.fxgraph_handler(gm, self)
        self.subgraph_idx += 1
        print("****Exit Custom Compiler****")
        return make_boxed_func(gm.forward)

    def compile_model(self):
        if self.run_custom_backend:
            if self.use_pytorch_ir:
                compiled_model = torch.compile(self.model, backend=self._custom_pytorch_compiler, dynamic=True, fullgraph=True)
            else:
                compiled_model = torch.compile(
                    self.model,
                    backend=aot_autograd(fw_compiler=self._custom_aten_compiler),
                    fullgraph=True,
                    dynamic=True
                )
        else:
            compiled_model = torch.compile(self.model)
        self.model = compiled_model

    # From here down, we have the code for the various jobs that can be run on the model.

    def run_fwbw_pass(self):
        self.compile_model()
        output = self.model(self.sample_input)
        torch.cuda.synchronize()
        loss = self.loss_fn(output, self.sample_label)

        del output
        loss.backward()
        torch.cuda.synchronize()

    def run_fw_pass(self):
        self.compile_model()
        self.model(self.sample_input)

    def run_eager_fwbw_pass(self):
        output = self.model(self.sample_input)
        torch.cuda.synchronize()
        output.sum().backward()
        torch.cuda.synchronize()

    def collect_postexecution_graph(self, dirname: str, name: str):
        from torch.profiler import ExecutionTraceObserver, profile

        rank = os.environ["RANK"]
        if "COMPILE" in os.environ and os.environ["COMPILE"] == "True":
            print("compile model")
            self.model = torch.compile(self.model)

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
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=0, warmup=10, active=30),
            on_trace_ready=kineto_trace_handler,
            with_flops=True,
        ) as prof:
            for epoch in range(20):
                if epoch == 19:
                    et.stop()
                if epoch == 7:
                    et.start()
                output = self.model(self.sample_input)
                torch.cuda.synchronize()
                prof.step()
                output.sum().backward()
                torch.cuda.synchronize()
                prof.step()
        et.stop()
        et.unregister_callback()

    def run_nsys_workload(self):  # noqa: C901. Ignore complaints about code being too complex.
        if "COMPILE" in os.environ and os.environ["COMPILE"] == "True":
            print("compile model")
            self.model = torch.compile(self.model)
        nb_iters = 10
        warmup_iters = 5
        for i in range(nb_iters):
            if os.environ["RANK"] == "0":
                print(f"start iteration {i}")
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
