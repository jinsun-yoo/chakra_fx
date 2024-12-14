import os
from typing import List

import torch.fx
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd


class ModelProfiler:
    def __init__(
        self,
        fxgraph_handler,
        use_pytorch_ir,
        model,
        sample_input,
        run_custom_backend_all_rank=False,
    ):
        self.rank = int(os.environ.get("RANK", 0))
        self.size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        "If true, work on PyTorch FX Graph, if false, work on aten FX Graph"
        self.use_pytorch_ir = use_pytorch_ir
        "Index of subgraph at each graph break. Increments with each subgraph"
        self.subgraph_idx = 0
        "This name will be used at, e.g. start of generated filenames"
        if not hasattr(self, "name"):
            print("Name not provided. Use default name.")
            self.name = "modelProfiler"
        "Unless otherwise noted, do not run custom backends, apart from rank 0."
        self.run_custom_backend = False
        if run_custom_backend_all_rank or self.local_rank == 0:
            self.run_custom_backend = True

        torch.cuda.set_device(int(self.local_rank))
        self.fxgraph_handler = fxgraph_handler
        self.model = model.cuda(int(self.local_rank))
        self.sample_input = sample_input

        return

    def convert_to_chakra(self, gm: torch.fx.GraphModule, exp_tag: str):
        from src.chakra_fx.passes.chakra_converter import ChakraConverter

        # TODO: We create one ChakraConverter per subgraph, but might have to change this due to DDP.
        # (Depends. There is a possibility no graph break is needed for DDP.)
        self.chakra_converter = ChakraConverter(self.name, self.subgraph_idx, exp_tag)
        self.chakra_converter.convert_to_chakra(gm)

    """Runs a training session, implemented by each profiler"""

    def run_training_session(self):
        print(f"Training session not implemented for profiler {self.name}")
        return

    """Simply complies a model & with the 'fxgraph_handler' backend.
    Inference on sample_input is used to trigger JIT compiling"""

    def run_sample_input(self):
        self.compile_model()
        self.model(self.sample_input)

    def run_inductor(self):
        self.run_custom_backend = False
        self.compile_model()
        self.model(self.sample_input)

    def __custom_pytorch_compiler(self, gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        self.fxgraph_handler(gm, self)
        self.subgraph_idx += 1
        return gm.forward

    def __custom_aten_compiler(self, gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        self.fxgraph_handler(gm, self)
        self.subgraph_idx += 1
        print("****Exit Custom Compiler****")
        return make_boxed_func(gm.forward)

    def compile_model(self):
        if self.run_custom_backend:
            if self.use_pytorch_ir:
                model = torch.compile(self.model, backend=self.__custom_pytorch_compiler)
            else:
                model = torch.compile(
                    self.model,
                    backend=aot_autograd(fw_compiler=self.__custom_aten_compiler),
                )
        else:
            model = torch.compile(self.model)
        self.model = model

    def run_eager(self):
        num_range = 1
        if "NUM_RANGE" in os.environ:
            num_range = int(os.environ["NUM_RANGE"])
        if os.environ["RANK"] == "0":
            print(f"num_range is {num_range}")
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

    def run_nsys_workload(self): # noqa: C901. Ignore complaints about code being too complex.
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
