import os

from torch.distributed import destroy_process_group

from src.args import parse_args
from src.chakra_fx.utils.time_recorder import timer

"""
    Usage: torchrun \
        --nproc-per-node=8 \
        --rdzv_id=... \
        profile_fxgraph.py \
"""

if __name__ == "__main__":
    if os.environ.get("RANK") == "0":
        timer.mark("program_start")
    args = parse_args()

    "Choose which profiler to use"
    model_name = args.model
    match model_name:
        case "llama":
            from src.chakra_fx.profilers.llama_profiler import LlamaProfiler

            profiler = LlamaProfiler(
                job=args.job,
                exp_tag=args.exp_tag,
                fxgraph_actions=args.action_list,
                run_custom_backend_all_rank=args.custom_backend_all_rank,
                dse_config_filepath=args.dse_config_filepath,
                sequential_generation=args.sequential_generation,
                use_cache=args.use_cache,
            )
        case "simple":
            from src.chakra_fx.profilers.simple_profiler import SimpleModelProfiler

            profiler = SimpleModelProfiler(
                job=args.job,
                exp_tag=args.exp_tag,
                fxgraph_actions=args.action_list,
                run_custom_backend_all_rank=args.custom_backend_all_rank,
                dse_config_filepath=args.dse_config_filepath,
                use_cache=args.use_cache,
            )
        case "nanogpt":
            from src.chakra_fx.profilers.nanogpt_profiler import NanoGptProfiler

            profiler = NanoGptProfiler(
                job=args.job,
                exp_tag=args.exp_tag,
                fxgraph_actions=args.action_list,
                run_custom_backend_all_rank=args.custom_backend_all_rank,
                dse_config_filepath=args.dse_config_filepath,
                use_cache=args.use_cache,
            )
        case "resnet18":
            from src.chakra_fx.profilers.resnet18_profiler import ResNetProfiler

            profiler = ResNetProfiler(
                job=args.job,
                exp_tag=args.exp_tag,
                fxgraph_actions=args.action_list,
                run_custom_backend_all_rank=args.custom_backend_all_rank,
                dse_config_filepath=args.dse_config_filepath,
                use_cache=args.use_cache,
            )

    # Run the job specified in args.
    # Note, some of these jobs, such as postexec_chakra, will not be torch.compiled.
    # That is, the custom backend compiler will not be triggered.
    if args.job == "fw":
        profiler.run_fw_pass()
    elif args.job == "fwbw":
        profiler.run_fwbw_pass()
    elif args.job == "postexec_chakra":
        profiler.collect_postexecution_graph(args.exp_tag, "trace")
    elif args.job == "nsys":
        profiler.run_nsys_workload()
    elif args.job == "eager":
        profiler.run_eager_fwbw_pass()
    if os.environ.get("RANK") == "0":
        timer.mark("program_end")
        timer.display_results()

    destroy_process_group()
