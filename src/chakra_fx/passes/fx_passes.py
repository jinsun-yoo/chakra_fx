import csv 
import os
import torch 

from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.utils.flop_counter import FlopCounterMode


def convert_save_chakra_graph(gm: torch.fx.GraphModule, profiler: ModelProfiler, exp_tag: str):
    if not os.path.exists(f"./{exp_tag}"):
        print(f"Output path {exp_tag} does not exist!")
        exit()
    profiler.convert_to_chakra(gm, exp_tag)

"Generates a dot file for this subgraph"
def save_dotfile(gm: torch.fx.GraphModule, profiler: ModelProfiler):
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_dot(
        f"{profiler.name}_PyTorch_{profiler.use_pytorch_ir}_subgraphidx_{profiler.subgraph_idx}_rank_{profiler.rank}.dot"
    )


"Generates a pdf file for this subgraph"
def save_pdffile(gm: torch.fx.GraphModule, profiler: ModelProfiler):
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_pdf(
        f"{profiler.name}_PyTorch_{profiler.use_pytorch_ir}_subgraphidx_{profiler.subgraph_idx}_rank_{profiler.rank}.pdf"
    )

"Prints the graph in tabular format to stdio. Haven't found how to forward to a file other than piping at command line"
def print_tabular_graph(gm: torch.fx.GraphModule):
    print(gm.graph.print_tabular())

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
    with open(f"{profiler.name}_histogram_rank_{profiler.rank}.csv", "a", newline="") as f:
        w = csv.writer(f)
        for op, count in ops_map.items():
            w.writerow([op, count["count"], count["op"]])


def get_operation_count(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(node)
        if success:
            with FlopCounterMode() as flop_counter_mode:
                node.target(*args, **kwargs)
                counted_flops = flop_counter_mode.get_total_flops()
                print(f"Counted flops for {node.target.__name__} is {counted_flops}")

"Simple function to check if backend has been called"
def just_hello(_: torch.fx.GraphModule, profiler: ModelProfiler):
    print(f"backend compiler has been called at rank {profiler.rank} for subgraph {profiler.subgraph_idx}")
    return