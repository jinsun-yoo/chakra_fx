import os
from typing import List

import torch
import torch.fx
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from src.chakra_fx.passes.chakra_converter import ChakraConverter
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
        sequential_generation: bool = False,
        combine_fx_subgraphs: bool = False,
        graph_passes: List[str] = None,
    ):
        self.rank = int(os.environ.get("RANK", 0))
        self.size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.exp_tag = exp_tag
        self.filename = f"{exp_tag}/syncfile.txt"
        self.sequential_generation = sequential_generation

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

        self.chakra_converter = ChakraConverter(
            name=self.name,
            dir_name=exp_tag,
            combine_fx_subgraphs=combine_fx_subgraphs,
            graph_passes=graph_passes,
        )
        return

    def _poll_start(self, filename):
        import time

        self.filename = filename
        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            if os.path.exists(self.filename):
                os.remove(self.filename)
            with open(self.filename, "w") as f:
                f.write("start\n")
            return

        while True:
            prev_rank = self.rank - 1
            # Sanity check. Skip rank 0.
            if prev_rank < 0:
                break
            if not os.path.exists(self.filename):
                time.sleep(1)
                continue
            with open(self.filename, "r") as f:
                if f"Rank {prev_rank} done." in f.read():
                    print(f"Rank {prev_rank} done. Proceeding with rank {self.rank}.")
                    break
            time.sleep(1)

    def _signal_end(self):
        if not self.sequential_generation:
            return
        with open(self.filename, "a") as f:
            f.write(f"Rank {self.rank} done.\n")

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
                compiled_model = torch.compile(
                    self.model,
                    backend=self._custom_pytorch_compiler,
                    dynamic=True,
                    fullgraph=True,
                )
            else:
                compiled_model = torch.compile(
                    self.model,
                    backend=aot_autograd(fw_compiler=self._custom_aten_compiler),
                    fullgraph=True,
                )
        else:
            compiled_model = torch.compile(self.model)
        self.model = compiled_model

    # From here down, we have the code for the various jobs that can be run on the model.

    def run_fwbw_pass(self):
        self.compile_model()
        output = self.model(self.sample_input)
        self.chakra_converter.finalize()
        exit()
        # TODO: With the call to self.model, already trace
        # Both FW and BW. Can exit here.
        # However, for actions other than FX->Chakra conversion, need to run BW as well.
        # Might need the code below.
        torch.cuda.synchronize()
        loss = self.loss_fn(output, self.sample_label)

        del output
        loss.backward()
        torch.cuda.synchronize()

    def run_fw_pass(self):
        self.compile_model()
        self.model(self.sample_input)
        self.chakra_converter.finalize()

    def run_eager_fwbw_pass(self):
        output = self.model(self.sample_input)
        torch.cuda.synchronize()
        output.sum().backward()
        torch.cuda.synchronize()

    def collect_postexecution_graph(self, dirname: str, name: str):
        from torch.profiler import ExecutionTraceObserver, profile
        
        dirname = os.path.join("./runs", dirname)

        rank = os.environ["RANK"]
        if "COMPILE" in os.environ and os.environ["COMPILE"] == "True":
            print("compile model")
            self.model = torch.compile(self.model)

        # for param in self.model.parameters():
        #     print(param.dtype)
        #     break
        self.model.to(dtype=torch.bfloat16)
        self.model.to_empty(device=f"cuda:{self.local_rank}")
        # for param in self.model.parameters():
        #     print(param.dtype)
        #     break
        # self.sample_input = torch.empty(self.sample_input.shape, device=f"cuda:{self.local_rank}", dtype=torch.bfloat16)

        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

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
            schedule=torch.profiler.schedule(wait=0, warmup=10, active=1),
            on_trace_ready=kineto_trace_handler,
            with_flops=True,
            execution_trace_observer=et,
        ) as prof:
            for _epoch in range(11):
                output = self.model(self.sample_input)
                torch.cuda.synchronize()
                output.sum().backward()
                torch.cuda.synchronize()
                prof.step()
        et.unregister_callback()
        # torch.distributed.destroy_process_group()

    def run_nsys_workload(self):
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
            if i >= warmup_iters:
                # push range for current iteration
                torch.cuda.nvtx.range_push("iteration{}".format(i))
                # push range for forward
                torch.cuda.nvtx.range_push("forward")

            output = self.model(self.sample_input)
            torch.cuda.synchronize()
            if i >= warmup_iters:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("backward")

            output.sum().backward()
            if i >= warmup_iters:
                torch.cuda.nvtx.range_pop()
                # pop iteration range
                torch.cuda.nvtx.range_pop()

            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
