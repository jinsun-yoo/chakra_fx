import os
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from typing import List 
from torch._dynamo.backends.common import aot_autograd

class ModelProfiler():
    def __init__(self, fxgraph_handler, use_pytorch_ir, model, sample_input, run_custom_backend_all_rank = False):
        self.rank = int(os.environ.get("RANK", 0))
        self.size = int(os.environ.get("WORLD_SIZE", 1))
        
        "If true, work on PyTorch FX Graph, if false, work on aten FX Graph"
        self.use_pytorch_ir = use_pytorch_ir
        "Index of subgraph at each graph break. Increments with each subgraph"
        self.subgraph_idx = 0
        "This name will be used at, e.g. start of generated filenames"
        if not hasattr(self, 'name'):
            print("Name not provided. Use default name.")
            self.name = "modelProfiler"
        "Unless otherwise noted, do not run custom backends, apart from rank 0."
        self.run_custom_backend = False
        if run_custom_backend_all_rank or self.rank == 0:
            self.run_custom_backend = True

        torch.cuda.set_device(int(self.rank))
        self.fxgraph_handler = fxgraph_handler
        self.model = model.cuda(int(self.rank))
        self.sample_input = sample_input

        return
    
    def convert_to_chakra(self, gm: torch.fx.GraphModule):
        from chakra_converter import ChakraConverter
        # TODO: We create one ChakraConverter per subgraph, but might have to change this due to DDP. 
        # (Depends. There is a possibility no graph break is needed for DDP.)
        self.chakra_converter = ChakraConverter(self.name, self.subgraph_idx)
        self.chakra_converter.convert_to_chakra(gm)

    """Runs a training session, implemented by each profiler"""
    def run_training_session(self):
        print(f"Training session not implemented for profiler {self.name}")
        return 
    
    """Simply complies a model & with the 'fxgraph_handler' backend. Inference on sample_input is used to trigger JIT compiling"""
    def run_sample_input(self):
        self.compile_model()
        self.model(self.sample_input)

    def __custom_pytorch_compiler(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.fxgraph_handler(gm, self)
        self.subgraph_idx += 1
        return gm.forward
    
    def __custom_aten_compiler(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.fxgraph_handler(gm, self)
        self.subgraph_idx += 1
        return make_boxed_func(gm.forward)

    def compile_model(self):
        if self.run_custom_backend:
            if self.use_pytorch_ir:
                model = torch.compile(self.model, backend=self.__custom_pytorch_compiler)
            else:
                model = torch.compile(self.model, backend=aot_autograd(fw_compiler=self.__custom_aten_compiler))
        else:
            model = torch.compile(self.model)
        self.model = model
