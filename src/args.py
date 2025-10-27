import argparse
from datetime import datetime, timezone


def parse_args():
    # Create parser
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        "--custom_backend_all_rank",
        type=bool,
        default=True,
        help="If true, run custom backend on all rank, not just 0",
    )
    parser.add_argument(
        "--fxgraph_actions",
        type=str,
        default="just",
        required=False,
        help="""Choose which actions to perform on the fxgraph provided by torch.compile. Comma delimited string.
        Refer to custom_backend_compiler for a detailed description of each action.""",
        choices=["chakra", "just", "pdf", "dot", "dumpgraph", "table", "histogram", "opcount", "print"],
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
        default="fwbw",
        required=False,
        help=(
            "What job to perform on the selected model. Possible options are:\n"
            "  fw: Run a single forward pass on a sample input.\n"
            "  fwbw: Run a single forward-backward pass on a sample input.\n"
            "  postexec_chakra: Collect the post-execution Chakra graph.\n"
            "  eager: Run a single forward-backward pass on a sample input, but using eager mode.\n"
            "  nsys: Run the model under nsys profiler."
        ),
        choices=["fw", "fwbw", "postexec_chakra", "eager", "nsys"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        required=False,
        help="""Choose which model to work on. Each model has its own profiler under src/chakra_fx/profilers.""",
        choices=["llama", "deepseek", "simple", "nanogpt", "resnet18"],
    )
    parser.add_argument(
        "--dse_config_filepath",
        type=str,
        default=None,
        required=False,
        help="[IGNORE FOR NOW] Filepath of the DSE configuration",
    )
    parser.add_argument(
        "--sequential_generation",
        action="store_true",
        help="If set to true, the processes will generate graphs one by one. For cases where all procs cannot fit in memory.",
    )
    parser.add_argument(
        "--use_real_device",
        default=False,
        required=False,
        action="store_true",
        help="If set to true, the model will be created on the real device. If false, will use the meta device",
    )
    parser.add_argument(
        "--combine_fx_subgraphs",
        default=False,
        required=False,
        action="store_true",
        help="If set to true, the forward and backward FX subgraphs will be combined.",
    )
    parser.add_argument(
        "--llama_config",
        type=str,
        default="debugmodel",
        required=False,
        help="""Choose which llama model config to use.""",
        choices=["debugmodel", "8B", "70B", "405B"],
    )
    parser.add_argument(
        "--graph_passes",
        type=str,
        default="",
        required=False,
        help="""Comma delimited string of graph passes to perform on FX graph before chakra conversion.
        E.g., 'bucket' to perform FSDP bucketing pass.""",
    )
    args = parser.parse_args()

    args.action_list = args.fxgraph_actions.split(",")
    args.graph_passes = args.graph_passes.split(",") if args.graph_passes else []
    if args.exp_tag == "":
        now_utc = datetime.now(timezone.utc)
        args.exp_tag = now_utc.strftime("%Y-%m-%d_%H-%M-%S")
    return args
