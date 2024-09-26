import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
import torch
import csv 
from torch.fx.passes.graph_drawer import FxGraphDrawer


from resnet18_profiler import ResNetProfiler
from nanogpt_profiler import NanoGptProfiler
from model_profiler import ModelProfiler

"Prints the graph in tabular format to stdio. Haven't found how to forward to a file other than piping at command line"
def print_tabular_graph(gm: torch.fx.GraphModule):
    print(gm.graph.print_tabular())

"Generates a dot file for this subgraph"
def print_dotfile(gm: torch.fx.GraphModule):
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_dot(f"{profiler.name}_PyTorch_{profiler.use_pytorch}_subgraphidx_{profiler.subgraph_idx}_rank_{profiler.rank}.dot")

"Generates a pdf file for this subgraph"
def print_pdffile(gm: torch.fx.GraphModule, profiler: ModelProfiler):
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_pdf(f"{profiler.name}_PyTorch_{profiler.use_pytorch}_subgraphidx_{profiler.subgraph_idx}_rank_{profiler.rank}.pdf")

"For aten graphs, save a histogram of target operators in a csv file. Ignores 'placeholder' ops." 
"For now, all subgraphs append to one csv file. subgraphs are split by 'output' in the csv file"
def get_aten_histogram(gm: torch.fx.GraphModule, profiler: ModelProfiler):
    ops_map = dict()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.target not in ops_map:
            ops_map[node.target] = {"count": 1, "op": node.op}
        else:
            value = ops_map[node.target]
            value["count"] = value["count"] + 1
            ops_map[node.target] = value
    with open(f'{profiler.name}_histogram_rank_{profiler.rank}.csv', 'a', newline="") as f:
        w = csv.writer(f)
        for op, count in ops_map.items():
            w.writerow([op, count["count"], count["op"]])

from torch.utils.flop_counter import FlopCounterMode
def get_operation_count(gm:torch.fx.GraphModule):
    for node in gm.graph.nodes:
        success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(node)
        if success:
            with FlopCounterMode() as flop_counter_mode:
                node.target(*args, **kwargs)
                counted_flops = flop_counter_mode.get_total_flops()
                print(f'Counted flops for {node.target.__name__} is {counted_flops}')
        
"Simple function to check if backend has been called"
def just_hello(gm: torch.fx.GraphModule):
    print("backend compiler has been called")
    return

"""
    Define the set of functions you want to use
"""
def my_compiler(gm: torch.fx.GraphModule, profiler: ModelProfiler):   
    #get_aten_histogram(gm, profiler) 
    #print_pdffile(gm, profiler)
    print_tabular_graph(gm)
    #just_hello(gm)

"""
    Usage: torchrun -nprocs-per-node=8 profile_fxgraph.py
"""
if __name__ == "__main__":
    "Choose which profiler to use"
    #profiler = NanoGptProfiler(my_compiler, True)
    profiler = ResNetProfiler(my_compiler, False)

    "Choose whether to only trigger JIT compile (through sample input), or running a training session"
    "Choose only one. For some reason running both glitches."
    #profiler.run_sample_input()
    profiler.run_training_session()
