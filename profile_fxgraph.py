from typing import List

import torch
import torch.fx
from functorch.compile import make_boxed_func
from torch.distributed import destroy_process_group

from src.args import parse_args
from src.chakra_fx.passes.fx_passes import (
    convert_save_chakra_graph,
    get_aten_histogram,
    get_operation_count,
    just_hello,
    print_tabular_graph,
    save_dotfile,
    save_fxgraph_module,
    save_pdffile,
)

"""
    Usage: torchrun \
        --nproc-per-node=8 \
        --rdzv_id=... \
        profile_fxgraph.py \
"""

if __name__ == "__main__":
    args = parse_args()

    # "Define the set of functions you want to use"
    def my_compiler(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        for action in args.action_list:
            match action:
                case "just":
                    just_hello(gm, 0)
                case "pdf":
                    save_pdffile(gm, args.exp_tag, "trace", 0)
                case "chakra":
                    convert_save_chakra_graph(gm, args.exp_tag, "trace", 0)
                case "dot":
                    save_dotfile(gm, args.exp_tag, "trace", 0)
                case "dumpgraph":
                    save_fxgraph_module(gm, args.exp_tag, "trace", 0)
                case "table":
                    print_tabular_graph(gm)
                case "histogram":
                    get_aten_histogram(gm, profiler)
                case "opcount":
                    get_operation_count(gm, profiler)
            exit()
        return make_boxed_func(gm.forward)

    "Choose which profiler to use"
    model = args.model
    if model == "simple":
        from src.chakra_fx.profilers.simple_profiler import SimpleModelProfiler

        profiler = SimpleModelProfiler(
            my_compiler,
            use_pytorch_ir=False,
            run_custom_backend_all_rank=args.custom_backend_all_rank,
            dse_config_filepath=args.dse_config_filepath,
        )
    elif model == "nanogpt":
        from src.chakra_fx.profilers.nanogpt_profiler import NanoGptProfiler

        profiler = NanoGptProfiler(
            my_compiler,
            use_pytorch_ir=False,
            run_custom_backend_all_rank=args.custom_backend_all_rank,
            dse_config_filepath=args.dse_config_filepath,
        )
    elif model == "llama":
        from src.chakra_fx.profilers.llama_profiler import LlamaProfiler

        profiler = LlamaProfiler(
            my_compiler,
            use_pytorch_ir=False,
            run_custom_backend_all_rank=args.custom_backend_all_rank,
            dse_config_filepath=args.dse_config_filepath,
            job=args.job,
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
    if args.job == "sample":
        profiler.run_sample_input()
    elif args.job == "inductor":
        profiler.run_inductor()
    elif args.job == "training":
        profiler.run_training_session()
    elif args.job == "postexec_chakra":
        profiler.collect_postexecution_graph(args.exp_tag, "trace")
    elif args.job == "nsys":
        profiler.run_nsys_workload()
    elif args.job == "eager":
        profiler.run_eager()

    destroy_process_group()
