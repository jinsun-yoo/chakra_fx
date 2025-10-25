from typing import List, Tuple

import torch
import torch.fx as fx

""" Splits the FX Node name into the operator and trailing index
    primals -> ('primals', 0)
    primals_1 -> ('primals', 1)
    primals_12 -> ('primals', 12)
    _unsafe_view -> ('_unsafe_view', 0)
    _unsafe_view_42 -> ('_unsafe_view', 42)
    _unsafe_view_2 -> ('_unsafe_view', 2)

    Returns a tuple (name_without_suffix, index). If no trailing
    numeric suffix is present (or the prefix before the underscore is
    empty), returns (original_string, 0).
"""


def split_trailing_multidigit(s: str) -> Tuple[str, int]:
    # find the last underscore
    idx = s.rfind("_")
    if idx == -1:
        return s, 0
    prefix = s[:idx]
    suffix = s[idx + 1 :]
    # suffix must be digits (allow single or multiple digits) and prefix
    # must be non-empty to be considered a valid split
    if prefix != "" and suffix.isdigit():
        return prefix, int(suffix)
    return s, 0


def combine_subgraphs(gm_list: List[torch.fx.GraphModule]) -> torch.fx.GraphModule:
    """
    Combine multiple FX subgraphs into a single FX graph.

    Args:
        gm_list (List[torch.fx.GraphModule]): List of FX GraphModules to combine.

    Returns:
        torch.fx.GraphModule: Combined FX GraphModule.
    """
    if len(gm_list) == 1:
        return gm_list[0]
    if len(gm_list) == 0 or len(gm_list) > 2:
        raise ValueError(f"Number of subgraphs should be 1 or 2: {len(gm_list)}")
    # Currently only support combining two subgraphs: fw and bw.
    fw_gm = gm_list[0]
    bw_gm = gm_list[1]

    fw_gm.graph.lint()
    fw_gm.graph.eliminate_dead_code()
    bw_gm.graph.lint()
    bw_gm.graph.eliminate_dead_code()
    # Combine the forward and backward graphs.
    combined_gm = torch.fx.GraphModule(fw_gm, torch.fx.Graph())
    combined_graph = combined_gm.graph
    node_map = {}
    op_id_counter = {}
    # Copy nodes from the forward graph.
    for node in fw_gm.graph.nodes:
        # use split_trailing_multidigit to extract base name and any index
        op_name, op_id = split_trailing_multidigit(node.name)
        # track the maximum seen op id for each op_name
        if op_name not in op_id_counter or op_id > op_id_counter[op_name]:
            op_id_counter[op_name] = op_id

        new_node = combined_graph.node_copy(node, lambda n: node_map[n])
        node_map[node] = new_node
        if new_node.op == "output":
            new_output_node = new_node
    # Copy nodes from the backward graph.
    for node in bw_gm.graph.nodes:
        # For bw pass, within fx graph, the op name counter start, resulting in same names (e.g. add_0 in both fw and bw)
        # This messes up with codegen, etc.
        # Hence, for ops in bw graph, restart numbering.
        op_name, _ = split_trailing_multidigit(node.name)
        # restart numbering for backward graph ops by incrementing the counter
        if op_name not in op_id_counter:
            op_id_counter[op_name] = 0
            new_name = op_name
        else:
            new_id = op_id_counter[op_name] + 1
            op_id_counter[op_name] = new_id
            new_name = f"{op_name}_{new_id}"
        new_node = combined_graph.node_copy(node, lambda n: node_map[n])
        new_node.name = new_name
        if type(node.target) is str:
            new_node.target = new_name
        node_map[node] = new_node
        if new_node.op != "placeholder" and all(input_node.op == "placeholder" for input_node in new_node.all_input_nodes):
            num_args = len(new_node.all_input_nodes)
            new_node.insert_arg(num_args, new_output_node)
    combined_graph.lint()
    combined_gm.recompile()
    # print(combined_gm.graph)
    return combined_gm


def fsdp_bucketing(gm: fx.GraphModule):
    # from post_grad.py
    # if os.environ["RANK"] == "0":
    #     print(gm.code)

    gm.graph.eliminate_dead_code()
    import functools

    from torch._inductor.fx_passes.fsdp import bucket_fsdp_reduce_scatter

    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="post_grad_passes",
    )
    GraphTransformObserver(gm, "bucket_reduce_scatters").apply_graph_pass(
        lambda graph: bucket_fsdp_reduce_scatter(
            graph.owning_module,
            None,
        )
    )
    from torch._inductor.fx_passes.fsdp import bucket_fsdp_all_gather

    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="post_grad_passes",
    )
    GraphTransformObserver(gm, "bucket_all_gathers").apply_graph_pass(
        lambda graph: bucket_fsdp_all_gather(
            graph.owning_module,
            None,
        )
    )
    gm.graph.lint()
    gm.recompile()
    # if os.environ["RANK"] == "0":
    #     print(gm.code)
