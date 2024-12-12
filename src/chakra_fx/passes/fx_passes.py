import csv
import os

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.utils.flop_counter import FlopCounterMode
from torch.distributed.distributed_c10d import _world


def convert_save_chakra_graph(gm: torch.fx.GraphModule, dirname: str, filename: str, subgraph_idx: int):
    if not os.path.exists(f"{dirname}"):
        print(f"Output path {dirname} does not exist!")
        exit()

    from chakra_fx.src.chakra_fx.passes.chakra_converter import ChakraConverter

    chakra_converter = ChakraConverter(filename, subgraph_idx, dirname)
    chakra_converter.convert_to_chakra(gm)


"Generates a dot file for this subgraph"


def save_dotfile(gm: torch.fx.GraphModule, dirname: str, name: str, subgraph_idx: int):
    rank = os.environ["RANK"]
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_dot(f"{dirname}/{name}_subgraph_{subgraph_idx}_rank_{rank}.dot")


"Generates a pdf file for this subgraph"


def save_pdffile(gm: torch.fx.GraphModule, dirname: str, name: str, subgraph_idx: int):
    rank = os.environ["RANK"]
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_pdf(f"{dirname}/{name}_subgraph_{subgraph_idx}_rank_{rank}.pdf")


def save_fxgraph_module(gm: torch.fx.GraphModule, dirname: str, name: str, subgraph_idx: int):
    rank = os.environ["RANK"]
    gm.to_folder(f"{dirname}/{name}_subgraph_{subgraph_idx}_rank_{rank}")


"Prints the graph in tabular format to stdio. Haven't found how to forward to a file other than piping at command line"


def print_tabular_graph(gm: torch.fx.GraphModule):
    print(gm.graph.print_tabular())


"For aten graphs, save a histogram of target operators in a csv file. Ignores 'placeholder' ops."
"For now, all subgraphs append to one csv file. subgraphs are split by 'output' in the csv file"


def get_aten_histogram(gm: torch.fx.GraphModule, name: str, subgraph_idx: int, rank: int):
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
    with open(f"{name}_histogram_rank_{rank}.csv", "a", newline="") as f:
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


def just_hello(_: torch.fx.GraphModule, subgraph_idx: int):
    rank = os.environ["RANK"]
    print(f"backend compiler has been called at rank {rank} for subgraph {subgraph_idx}")
    print(_world.pg_group_ranks)
    print('printed group rank')
    return
