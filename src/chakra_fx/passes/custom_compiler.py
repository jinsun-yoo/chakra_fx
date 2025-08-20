from typing import TYPE_CHECKING, List

import torch
from torch.distributed import destroy_process_group

from src.chakra_fx.passes.fx_action import (
    convert_save_chakra_graph,
    get_aten_histogram,
    get_operation_count,
    just_hello,
    print_tabular_graph,
    save_dotfile,
    save_fxgraph_module,
    save_pdffile,
)

if TYPE_CHECKING:
    from src.chakra_fx.profilers.model_profiler import ModelProfiler


def _handle_action(action: str, gm: torch.fx.GraphModule, exp_tag: str, profiler: "ModelProfiler", called_before: bool, use_cache: int):
    """Handle a single action from the action list."""
    match action:
        case "chakra":
            filename = "trace"
            if called_before:
                filename = "trace_bw"
            convert_save_chakra_graph(gm, exp_tag, filename, 0, use_cache)
        case "just":
            just_hello(gm, 0)
        case "pdf":
            save_pdffile(gm, exp_tag, "trace", 0)
        case "dot":
            save_dotfile(gm, exp_tag, "trace", 0)
        case "dumpgraph":
            save_fxgraph_module(gm, exp_tag, "trace", 0)
        case "table":
            print_tabular_graph(gm)
        case "histogram":
            get_aten_histogram(gm, profiler)
        case "opcount":
            get_operation_count(gm, profiler)


# Assumption: No graph break apart from FW/BW
called_before = False


def build_custom_backend_compiler(action_list: List[str], exp_tag: str, profiler: "ModelProfiler"):
    # This is the custom backend compiler that torch.compile will call after parsing the FX graph.
    # The original intent of this interface is
    # for the custom 'backend compiler' to compile the graph (perform optimizations)
    # However, we use this to extract the Chakra Graph, or perform other actions.
    # We do not intend for the computation to actually take place.
    # Hence, we exit, instead of returning anything.
    def custom_backend_compiler(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        global called_before
        for action in action_list:
            _handle_action(action, gm, exp_tag, profiler, called_before, profiler.use_cache)
        # The assumption is that the custom compiler is called only once (i.e. there will be no graph break)
        # profiler._signal_end()
        if called_before:
            profiler._signal_end()
            destroy_process_group()
            exit()
        called_before = True
        # return make_boxed_func(gm.forward)

    return custom_backend_compiler
