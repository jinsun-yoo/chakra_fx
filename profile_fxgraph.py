import argparse
import csv
import os
from datetime import datetime, timezone
from typing import List

import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.utils.flop_counter import FlopCounterMode

from model_profiler import ModelProfiler

"Prints the graph in tabular format to stdio. Haven't found how to forward to a file other than piping at command line"


def print_tabular_graph(gm: torch.fx.GraphModule):
    print(gm.graph.print_tabular())


"Generates a dot file for this subgraph"


def print_dotfile(gm: torch.fx.GraphModule, profiler: ModelProfiler):
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_dot(
        f"{profiler.name}_PyTorch_{profiler.use_pytorch_ir}_subgraphidx_{profiler.subgraph_idx}_rank_{profiler.rank}.dot"
    )


"Generates a pdf file for this subgraph"


def print_pdffile(gm: torch.fx.GraphModule, profiler: ModelProfiler):
    g = FxGraphDrawer(gm, "graph")
    g.get_dot_graph().write_pdf(
        f"{profiler.name}_PyTorch_{profiler.use_pytorch_ir}_subgraphidx_{profiler.subgraph_idx}_rank_{profiler.rank}.pdf"
    )


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


def get_chakra_graph(gm: torch.fx.GraphModule, profiler: ModelProfiler, exp_tag: str):
    if not os.path.exists(f"./{exp_tag}"):
        print(f"Output path {exp_tag} does not exist!")
        exit()
    profiler.convert_to_chakra(gm, exp_tag)


"Simple function to check if backend has been called"


def just_hello(_: torch.fx.GraphModule, profiler: ModelProfiler):
    print(f"backend compiler has been called at rank {profiler.rank} for subgraph {profiler.subgraph_idx}")
    return


"""
    Usage: torchrun --nproc-per-node=8 profile_fxgraph.py
"""


def parse_args():
    # Create parser
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        "--custom_backend_all_rank",
        type=bool,
        default=False,
        help="If true, run custom backend on all rank, not just 0",
    )
    parser.add_argument(
        "--dse_config_filepath",
        type=str,
        default=None,
        required=False,
        help="Filepath of the DSE configuration",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="just",
        required=False,
        help="""Comma delimited key strings of which functions to invoke in backend compiler.
        'just': just_hello, 'chakra': get_chakra_graph, 'pdf': print_pdf_file""",
    )
    parser.add_argument(
        "--exp_tag",
        type=str,
        default="",
        required=False,
        help="""A string(tag) to uniquely identify this experiment. Will be used for output directory name, etc.
        Default is 'YYYY-MM-SS_HH-MM-SS' (UTC)""",
    )
    parser.add_argument(
        "--job",
        type=str,
        default="sample",
        required=False,
        help="""Either 'sample' or 'training'.
        \nDecides whether to run a single forward pass on a sample input, or a full training session.""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nanogpt",
        required=False,
        help="""Either 'nanogpt' or 'resnet18'. Chooses which model to work on.""",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    action_list = args.actions.split(",")
    exp_tag = args.exp_tag
    if exp_tag == "":
        now_utc = datetime.now(timezone.utc)
        exp_tag = now_utc.strftime("%Y-%m-%d_%H-%M-%S")

    "Define the set of functions you want to use"

    def my_compiler(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        for action in action_list:
            match action:
                case "just":
                    just_hello(gm, profiler)
                case "pdf":
                    print_pdffile(gm, profiler)
                case "chakra":
                    get_chakra_graph(gm, profiler, exp_tag)
                case "dot":
                    print_dotfile(gm, profiler)
                case "table":
                    print_tabular_graph(gm)
                case "histogram":
                    get_aten_histogram(gm, profiler)
                case "opcount":
                    get_operation_count(gm, profiler)

        print("***EXITING!!***")
        exit()

    "Choose which profiler to use"
    model = args.model
    if model == "nanogpt":
        from nanogpt_profiler import NanoGptProfiler

        profiler = NanoGptProfiler(
            my_compiler,
            use_pytorch_ir=False,
            run_custom_backend_all_rank=args.custom_backend_all_rank,
            dse_config_filepath=args.dse_config_filepath,
        )
    elif model == "resnet18":
        from resnet18_profiler import ResNetProfiler

        profiler = ResNetProfiler(
            my_compiler,
            use_pytorch_ir=False,
            run_custom_backend_all_rank=args.custom_backend_all_rank,
        )

    "Choose whether to only trigger JIT compile (through sample input), or running a training session"
    "Choose only one. For some reason running both glitches."
    # TODO: Remove invalid values
    job = args.job
    if job == "sample":
        profiler.run_sample_input()
    elif job == "training":
        profiler.run_training_session()
