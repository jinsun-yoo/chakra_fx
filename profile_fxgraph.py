import argparse
from datetime import datetime, timezone
from typing import List

import torch
import torch.fx

from src.chakra_fx.passes.fx_passes import print_tabular_graph, save_pdffile, save_dotfile, convert_save_chakra_graph, just_hello, get_aten_histogram, get_operation_count
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
                    save_pdffile(gm, profiler)
                case "chakra":
                    convert_save_chakra_graph(gm, profiler, exp_tag)
                case "dot":
                    save_dotfile(gm, profiler)
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
        from src.chakra_fx.profilers.nanogpt_profiler import NanoGptProfiler

        profiler = NanoGptProfiler(
            my_compiler,
            use_pytorch_ir=False,
            run_custom_backend_all_rank=args.custom_backend_all_rank,
            dse_config_filepath=args.dse_config_filepath,
        )
    elif model == "resnet18":
        from src.chakra_fx.profilers.resnet18_profiler import ResNetProfiler

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
