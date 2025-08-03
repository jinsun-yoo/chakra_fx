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
        'just': just_hello, 'chakra': get_chakra_graph, 'pdf': print_pdf_file, 'dumpgraph': save_fxgraph_module""",
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
        help="""Either 'sample' or 'training' or 'postexec_chakra.
        \nDecides whether to run a single forward pass on a sample input, or a full training session.""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nanogpt",
        required=False,
        help="""Either 'simple' or 'nanogpt' or 'resnet18'. Chooses which model to work on.""",
    )
    args = parser.parse_args()

    args.action_list = args.actions.split(",")
    if args.exp_tag == "":
        now_utc = datetime.now(timezone.utc)
        args.exp_tag = now_utc.strftime("%Y-%m-%d_%H-%M-%S")
    return args
