import os

import torch
import torch._inductor.fx_utils as fx_utils
import torch.distributed as dist
import torch.fx as fx
from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
    Int64List,
)
from chakra.schema.protobuf.et_def_pb2 import (
    AttributeProto as ChakraAttr,
)
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from chakra.schema.protobuf.et_def_pb2 import (
    NodeType as ChakraNodeType,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.utils.flop_counter import FlopCounterMode

# Map from c10d operator string to Chakra collective enumeration
c10d_chakra_map = {
    "all_reduce": ALL_REDUCE,
    "all_gather": ALL_GATHER,
    "all_gather_into_tensor": ALL_GATHER,
    "all_to_all": ALL_TO_ALL,
    "reduce_scatter": REDUCE_SCATTER,
    "reduce_scatter_tensor": REDUCE_SCATTER,
    "broadcast": BROADCAST,
}

# Operators that we will skip, because they either do nothing meaningful compute wise, or are simple placeholders.
# for (c10d::)wait_tensor, the dependency in Chakra graph already implies that the communication has to finish.
# No need to add a 'wait' element.
# TODO: Async comms?
# TODO: Are these operators _really_ not meaningful compute-wise? Need to check
skip_operator_list = ["wait_tensor", "view", "t", "transpose", "split"]

timestamp_map = {
    "linear": {
        # Kineto trace of actual execution
        1024 * 12288 * 6144: 1531,  # .677,
        1024 * 12288 * 49152: 9453,
    }
}


# Functions to help inspecting FX Nodes.
def fxnode_seqnr(fx_node: fx.Node):
    """Get the sequence number of the FX Node."""
    if fx_node.name == "root":
        return -1
    if fx_node.name == "output":
        return -2
    if fx_node.op == "placeholder":
        return fx_node.name
    if "seq_nr" not in fx_node.meta:
        print(f"Node name {fx_node.name} does not have seq_nr")
        return fx_node.name
    return fx_node.meta["seq_nr"]


def node_debug_id_str(fx_node: fx.Node):
    """Human readable string to help identify each node in debug."""
    return f"Node name:'{fx_node.name}', seq_id:{fxnode_seqnr(fx_node)}"


def is_comm_node(fx_node: fx.Node):
    target: OpOverload = fx_node.target
    return target.namespace == "_c10d_functional"


def is_comp_node(fx_node: fx.Node):
    target: OpOverload = fx_node.target
    return target.namespace == "aten"


class ChakraConverter:
    def __init__(self, name: str, subgraph_idx: int, dir_name: str):
        self.name = name
        self.subgraph_idx = subgraph_idx
        subgraphstr = ""
        if self.subgraph_idx > 0:
            subgraphstr = f"_subgraph-idx_{self.subgraph_idx}"
        if dir_name != "":
            dir_name += "/"
        self.filename = f"{dir_name}{self.name}{subgraphstr}.{dist.get_rank()}.et"

        # Incremented whenever Chakra Node is crated
        self.chakra_node_id = 0
        # TODO: Unverified assumption: FXNode.name is unique (sanity check in record_fx_node)
        # Very likely unique, since nodes with the same operator are automatically named 'addmm_1', 'addmm_2', etc.
        # Used to check if a FX Node of a spefic sequence id has been seen.
        self.fxnode_name_lookup_map = {}
        # Filld only when an FX Node has been converted to a Chakra Node.
        self.fxname_chakraid_map = {}

    def create_chakra_node(self, node_name: str, node_type: ChakraNodeType) -> ChakraNode:
        """Generate a new ChakraNode with a unique ID."""
        node = ChakraNode()
        node.id = self.chakra_node_id
        node.name = node_name
        node.type = node_type
        self.chakra_node_id += 1
        return node

    def create_comm_node(self, fx_node: fx.Node) -> ChakraNode:
        node_name = fx_node.name
        op_name = fx_node.target._opname
        comm_size = 0
        comm_type = c10d_chakra_map[op_name]
        if "val" not in fx_node.meta or not isinstance(fx_node.meta["val"], FakeTensor):
            print(f"Sanity check: {node_debug_id_str(fx_node)} is c10d, but fake output is not found")
            exit()

        import torch.distributed.distributed_c10d as c10d

        process_group_name = fx_node.args[-1]
        process_group_ranks = c10d.get_process_group_ranks(c10d._resolve_process_group(process_group_name))
        num_process_groups = len(c10d._world.pg_names)

        # Use FakeTensor, which is included in the FX Graph as a fake input, to determine communication size
        comm_tensor: FakeTensor = fx_node.meta["val"]
        tensor_dtype = comm_tensor.element_size()
        tensor_numelements = comm_tensor.numel()
        comm_size = tensor_dtype * tensor_numelements

        if comm_type == ALL_GATHER or REDUCE_SCATTER:
            # The fx_node, which is the 'result' of the ALL_GATHER/REDUCE_SCATTER, points to the *output* tensor.
            # Therefore, we have to divide it by # of ranks to get input tensor size.
            comm_size = int(comm_size / len(process_group_ranks))

        chakra_node = self.create_chakra_node(node_name, COMM_COLL_NODE)
        chakra_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        chakra_node.attr.append(ChakraAttr(name="comm_type", int64_val=c10d_chakra_map[op_name]))
        chakra_node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))

        # The ProcessGroup related attribute name and values follow the proposal in the MLC Chakra WG meeting of 2024-09-09.
        # The actual attribute names may change in the future.
        chakra_node.attr.append(ChakraAttr(name="pg_name", string_val=process_group_name))
        chakra_node.attr.append(ChakraAttr(name="group_size", int64_val=len(process_group_ranks)))
        chakra_node.attr.append(ChakraAttr(name="group_count", int64_val=num_process_groups))
        pg_ranks_protobuf = Int64List()
        pg_ranks_protobuf.values.extend(process_group_ranks)
        chakra_node.attr.append(ChakraAttr(name="ranks", int64_list=pg_ranks_protobuf))
        return chakra_node

    def estimate_flop_count(self, fx_node: fx.Node) -> int:
        success, args, kwargs = fx_utils.get_fake_args_kwargs(fx_node)
        if not success:
            # TODO: Add logger, and make this print only in verbose mode.
            # print(f"{node_debug_id_str(fx_node)} has flop not countable operator: {fx_node.target._opname}") # noqa: ERA001
            return 1000  # Arbitrary number

        with FlopCounterMode(display=False) as flop_counter_mode:
            if fx_node.target._overloadpacket not in flop_counter_mode.flop_registry:
                if os.environ["RANK"] == "0":
                    print(
                        f"{node_debug_id_str(fx_node)} has operator out of the registry: {fx_node.target._opname}, {fx_node.target._overloadpacket}"
                    )
                return 1000  # Arbitrary number
            fx_node.target(*args, **kwargs)
            return flop_counter_mode.get_total_flops()

    def estimate_tensor_size(self, fx_node: fx.Node) -> int:
        success, args, kwargs = fx_utils.get_fake_args_kwargs(fx_node)
        if not success:
            if os.environ["RANK"] == "0":
                print(
                    f"{node_debug_id_str(fx_node)} has estimated flopcount but no tensor_size: {fx_node.target._opname}"
                )
            return 1000  # Arbitrary number

        a = args[0].size()
        b = args[1].size()

        aten = torch.ops.aten
        if fx_node.target._overloadpacket != aten.mm:
            a = args[1].size()
            b = args[2].size()
        # TODO: Size
        size = 4
        estimated_tensor_size = 2 * size * (a[0] * a[1] + b[0] * b[1] + a[0] * b[1])
        if os.environ["RANK"] == "0":
            print(f"Estimated tensor size, a: {a[0]} {a[1]} b: {b[0]} {b[1]} result {estimated_tensor_size}")
        return estimated_tensor_size

    def lookup_duration(self, fx_node: fx.Node) -> int: # noqa: C901. TODO: Make logger print only for rank=0. (Remove the branches = code complexity)
        success, args, kwargs = fx_utils.get_fake_args_kwargs(fx_node)
        if not success:
            if os.environ["RANK"] == "0":
                print(
                    f"{node_debug_id_str(fx_node)} has estimated flopcount but no tensor_size: {fx_node.target._opname}"
                )
            return 1000  # Arbitrary number

        lookup_op = "-1"
        target_op = fx_node.target._overloadpacket
        aten = torch.ops.aten
        if target_op == aten.addmm or target_op == aten.mm:
            lookup_op = "linear"

        a = args[0].size()
        b = args[1].size()

        aten = torch.ops.aten
        if fx_node.target._overloadpacket != aten.mm:
            a = args[1].size()
            b = args[2].size()
        if a[1] != b[0]:
            print(f"{node_debug_id_str(fx_node)} tensor size does not match for matrix multiplication: {a[1]}, {b[0]}")

        numel = a[0] * a[1] * b[1]
        if lookup_op not in timestamp_map:
            if os.environ["RANK"] == "0":
                print(f"{node_debug_id_str(fx_node)} does not have lookup op {lookup_op} for {target_op} in lookup map")
            return -1
        if numel in timestamp_map[lookup_op]:
            lookup_duration = timestamp_map[lookup_op][numel]
        else:
            if os.environ["RANK"] == "0":
                print(
                    f"{node_debug_id_str(fx_node)} Estimated tensor size, a: {a[0]} {a[1]} b: {b[0]} {b[1]} with total numel {numel} not in map"
                )
            return -1
        if os.environ["RANK"] == "0":
            print(f"Estimated duration, a: {a[0]} {a[1]} b: {b[0]} {b[1]} result {lookup_duration}")
        return lookup_duration

    def measure_duration_microsecond(self, fx_node: fx.Node) -> int:
        import torch._subclasses.fake_tensor

        with maybe_disable_fake_tensor_mode():

            def realify_fake_tensor(arg) -> torch.Tensor:
                # "Scalar" value
                if type(arg) is fx.Node:
                    return arg
                fake_tensor: torch._subclasses.fake_tensor.FakeTensor = arg.meta["val"]
                real_tensor = torch.rand(fake_tensor.size(), dtype=fake_tensor.dtype, device=fake_tensor.device)
                return real_tensor

            flat_args = [realify_fake_tensor(arg) for arg in fx_node.args]
            flat_kwargs = fx_node.kwargs
            num_iters = 10
            import time

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            num_warmup_iters = 3
            for _ in range(num_warmup_iters):
                fx_node.target._overloadpacket(*flat_args, **flat_kwargs)
            """
            Ref: https://github.com/pytorch/pytorch/blob/45d62d6fc59e43e674985edd138e396268fe12fd/torch/distributed/_tools/runtime_estimator.py#L209
            torch.cuda.Event.record() inserts events into the GPU stream before/after running the kernel.
            This allows us to measure GPU time without host latency, etc.
            """
            cpu_start = time.time()
            start_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                fx_node.target._overloadpacket(*flat_args, **flat_kwargs)
            end_event.record(torch.cuda.current_stream())
            cpu_end = time.time()
            torch.cuda.synchronize()
            cpu_time = (cpu_end - cpu_start) * 1_000_000  # Second to microsecond
            if os.environ["RANK"] == "0":
                print(
                    f"For fx node {fx_node.name}, cpu measured is {cpu_time}, event dur is {start_event.elapsed_time(end_event)}"
                )
            total_op_time = start_event.elapsed_time(end_event) * 1000  # Millisecond to microsecond
            mean_op_time = total_op_time / num_iters
        return int(mean_op_time)

    def create_comp_node(self, fx_node: fx.Node) -> ChakraNode:
        node_name = fx_node.name
        estimated_flops = self.estimate_flop_count(fx_node)
        estimated_tensor_size = 0
        estimated_duration = 0
        if estimated_flops != 1000:
            estimated_tensor_size = self.estimate_tensor_size(fx_node)
            estimated_duration = self.measure_duration_microsecond(fx_node)

        if estimated_duration == -1:
            estimated_duration = 4000
        chakra_node = self.create_chakra_node(node_name, COMP_NODE)
        chakra_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        chakra_node.attr.append(ChakraAttr(name="num_ops", int64_val=estimated_flops))
        chakra_node.attr.append(ChakraAttr(name="tensor_size", uint64_val=estimated_tensor_size))
        chakra_node.duration_micros = estimated_duration
        return chakra_node

    def record_fx_node(self, fx_node: fx.Node):
        if fx_node.name in self.fxnode_name_lookup_map:
            # Sanity check. This is VERY unlikely.
            print(f"{node_debug_id_str(fx_node)} Already has a node with the same name!")
        self.fxnode_name_lookup_map[fx_node.name] = fx_node

    def add_to_chakra_graph(self, chakra_node: ChakraNode, fx_node: fx.Node):
        encode_message(self.et_file, chakra_node)
        self.fxname_chakraid_map[fx_node.name] = chakra_node.id

    # Find which Chakra nodes to declare as upstream dependency for this Chakra node.
    # Starting from corresponding FX node, iterate the FX Graph upwards
    # until we find nodes that have already been converted to Chakra Nodes.
    def add_upstream_dependency(self, chakra_node: ChakraNode, fx_node: fx.Node):
        upstream_search_queue = fx_node.all_input_nodes
        while len(upstream_search_queue) > 0:
            upstream_candidate = upstream_search_queue.pop()
            if upstream_candidate.name == "root" or upstream_candidate.op == "placeholder":
                continue
            upstream_name = upstream_candidate.name
            # Sanity check.
            # Because of topological ordering, we should have already recorded any upstream within the FX graph.
            if upstream_name not in self.fxnode_name_lookup_map:
                print(f"{node_debug_id_str(upstream_candidate)} not seen before!")
            if upstream_name in self.fxname_chakraid_map:
                upstream_chakra_id = self.fxname_chakraid_map[upstream_name]
                # Check if we already added this Chakra node as upstream dependency.
                if upstream_chakra_id not in chakra_node.data_deps:
                    chakra_node.data_deps.append(upstream_chakra_id)
                # Do not query any more upstream node of _this_ upstream_candidate FX Node.
                continue
            for next_upstream in upstream_candidate.all_input_nodes:
                upstream_search_queue.append(next_upstream)
        return chakra_node

    def convert_to_chakra(self, gm: fx.GraphModule):
        with open(self.filename, "wb") as et:
            self.et_file = et
            encode_message(et, GlobalMetadata(version="0.0.4"))
            for fx_node in gm.graph.nodes:
                # Record node info in internal lookup tables.
                self.record_fx_node(fx_node)

                # Skip nodes that are 1) placeholders 2) insignificant compute
                if fx_node.name == "root" or fx_node.op in ["placeholder", "output"] or "getitem" in fx_node.name:
                    continue
                # At this point, all remaining nodes should target the type OpOverload
                # OpOverload is a PyTorch wrapper for ATen/c10d operators, defined in torch/_ops.py
                # TODO: Complex cases may involve target with any Callable that are not OpOverload type.
                if not isinstance(fx_node.target, OpOverload):
                    print(
                        f"""{node_debug_id_str(fx_node)},
                        After filters, we still have a node whose target is not OpOverload but {type(fx_node.target)}"""
                    )
                    continue
                # Remove operators that does not do anything significant.
                elif fx_node.target._opname in skip_operator_list:
                    continue
                # Convert to Chakra node. Check if it should be converted to a COMP or a COMM node.
                elif is_comm_node(fx_node):
                    chakra_node = self.create_comm_node(fx_node)
                elif is_comp_node(fx_node):
                    chakra_node = self.create_comp_node(fx_node)
                else:
                    print(f"{node_debug_id_str(fx_node)} is neither c10d or aten")
                    continue
                chakra_node = self.add_upstream_dependency(chakra_node, fx_node)
                self.add_to_chakra_graph(chakra_node, fx_node)
