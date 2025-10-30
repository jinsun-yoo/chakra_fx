import copy
from typing import TYPE_CHECKING, List

import torch
from torch._inductor.compile_fx import make_boxed_func

from src.chakra_fx.passes.fx_action import (
    convert_save_chakra_graph,
    get_aten_histogram,
    get_operation_count,
    just_hello,
    print_graph_code,
    print_graph_to_file,
    print_tabular_graph,
    save_dotfile,
    save_fxgraph_module,
    save_pdffile,
)

if TYPE_CHECKING:
    from src.chakra_fx.profilers.model_profiler import ModelProfiler


def _handle_action(action: str, gm: torch.fx.GraphModule, exp_tag: str, profiler: "ModelProfiler"):
    """Handle a single action from the action list."""
    match action:
        case "chakra":
            # Avoid importing DeepseekProfiler (circular import). Use a runtime
            # class-name check instead of isinstance with the actual class.
            convert_save_chakra_graph(
                gm,
                profiler.chakra_converter,
                profiler.__class__.__name__ == "DeepseekProfiler",
            )
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
        case "print":
            print_graph_code(gm)
        case "file":
            print_graph_to_file(gm, exp_tag, "trace")


def build_custom_backend_compiler(action_list: List[str], exp_tag: str, profiler: "ModelProfiler"):
    # This is the custom backend compiler that torch.compile will call after parsing the FX graph.
    # The original intent of this interface is
    # for the custom 'backend compiler' to compile the graph (perform optimizations)
    # However, we use this to extract the Chakra Graph, or perform other actions.
    # We do not intend for the computation to actually take place.
    # Hence, we exit, instead of returning anything.
    def custom_backend_compiler(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
        for action in action_list:
            _handle_action(action, gm, exp_tag, profiler)
        # Avoid importing DeepseekProfiler at runtime to prevent circular
        # imports. Use class-name comparison as a lightweight runtime check.
        if profiler.__class__.__name__ != "DeepseekProfiler":
            return make_boxed_func(gm.forward)
        # operate on a deep copy to avoid mutating the original GraphModule
        return_gm = copy.deepcopy(gm)

        # Include the to_copy to graph, but remove from what we return, so that Dynamo can trace backward
        # without trying to actually execute to_copy
        for node in return_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten._to_copy.default:
                try:
                    replacement = node.args[0] if node.args else None
                    if replacement is not None:
                        node.replace_all_uses_with(replacement)
                    return_gm.graph.erase_node(node)
                    # ensure the GraphModule is up-to-date after modification
                    return_gm.graph.lint()
                    return_gm.recompile()
                except Exception:
                    # if anything goes wrong, skip removal for this node
                    pass
        return_gm.graph.lint()
        return_gm.recompile()
        # if os.getenv("RANK", "1") == "0":
        #     print(return_gm.code)

        return make_boxed_func(return_gm.forward)
        # The assumption is that the custom compiler is called only once (i.e. there will be no graph break)
        # profiler._signal_end()
        # return make_boxed_func(gm.forward)

    return custom_backend_compiler
