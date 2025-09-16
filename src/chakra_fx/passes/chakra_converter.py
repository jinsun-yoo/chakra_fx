import os
import csv
import time
from typing import Tuple

import torch
import torch._inductor.fx_utils as fx_utils
import torch._subclasses.fake_tensor
import torch.distributed as dist
from torch.distributed import destroy_process_group
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
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily

# from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.utils.flop_counter import FlopCounterMode

from src.chakra_fx.utils.time_recorder import timer

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
    def __init__(self, name: str, subgraph_idx: int, dir_name: str, use_cache: int):
        self.name = name
        self.subgraph_idx = subgraph_idx
        subgraphstr = ""
        if self.subgraph_idx > 0:
            subgraphstr = f"_subgraph-idx_{self.subgraph_idx}"
        if dir_name != "":
            dir_name += "/"
        self.filename = f"{dir_name}{self.name}{subgraphstr}.{dist.get_rank()}.et"
        self.use_cache = use_cache

        # --- [MODIFIED] ---
        # 为缓存功能增加路径和内存字典
        self.csv_path = "/workspace/chakra_fx/gemm_collected.csv"
        self.duration_cache = {}
        # --- [END MODIFIED] ---

        # Incremented whenever Chakra Node is crated
        self.chakra_node_id = 0
        # TODO: Unverified assumption: FXNode.name is unique (sanity check in record_fx_node)
        # Very likely unique, since nodes with the same operator are automatically named 'addmm_1', 'addmm_2', etc.
        # Used to check if a FX Node of a spefic sequence id has been seen.
        self.fxnode_name_lookup_map = {}
        # Filld only when an FX Node has been converted to a Chakra Node.
        self.fxname_chakraid_map = {}

    # --- [NEW] ---
    # 新增一个方法, 用于从CSV文件中加载已有的duration数据到内存缓存中
    def _load_duration_cache(self):
        """Loads the duration cache from the CSV file."""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            try:
                next(reader)  # Skip header
                for row in reader:
                    if len(row) == 4:
                        _, a_shape_str, b_shape_str, duration_str = row
                        key = (a_shape_str, b_shape_str)
                        # 避免重复加载
                        if key not in self.duration_cache:
                            self.duration_cache[key] = int(duration_str)
            except StopIteration:
                # 文件为空, 什么也不做
                pass
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse row in cache file: {row}. Error: {e}")

    # --- [END NEW] ---

    def create_chakra_node(
        self, node_name: str, node_type: ChakraNodeType
    ) -> ChakraNode:
        """Generate a new ChakraNode with a unique ID."""
        node = ChakraNode()
        node.id = self.chakra_node_id
        node.name = node_name
        node.type = node_type
        self.chakra_node_id += 1
        return node

    def create_comm_node(self, fx_node: fx.Node, comm_setting: dict) -> ChakraNode:
        node_name = fx_node.name
        op_name = fx_node.target._opname
        comm_size = 0
        comm_type = c10d_chakra_map[op_name]
        if "val" not in fx_node.meta or not isinstance(fx_node.meta["val"], FakeTensor):
            print(
                f"Sanity check: {node_debug_id_str(fx_node)} is c10d, but fake output is not found"
            )
            exit()

        import torch.distributed.distributed_c10d as c10d

        process_group_name = fx_node.args[-1]
        process_group_ranks = c10d.get_process_group_ranks(
            c10d._resolve_process_group(process_group_name)
        )
        num_process_groups = len(c10d._world.pg_names)

        comm_setting.setdefault(str(process_group_name), process_group_ranks)

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
        chakra_node.attr.append(
            ChakraAttr(name="comm_type", int64_val=c10d_chakra_map[op_name])
        )
        chakra_node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))

        # The ProcessGroup related attribute name and values follow the proposal in the MLC Chakra WG meeting of 2024-09-09.
        # The actual attribute names may change in the future.
        chakra_node.attr.append(
            ChakraAttr(name="comm_group", string_val=process_group_name)
        )
        chakra_node.attr.append(
            ChakraAttr(name="group_size", int64_val=len(process_group_ranks))
        )
        chakra_node.attr.append(
            ChakraAttr(name="group_count", int64_val=num_process_groups)
        )
        pg_ranks_protobuf = Int64List()
        pg_ranks_protobuf.values.extend(process_group_ranks)
        chakra_node.attr.append(ChakraAttr(name="ranks", int64_list=pg_ranks_protobuf))
        return chakra_node

    # Obtain the flop count of an FX Node based on the recorded operation and (symbolic) tensor argument.
    # We use the 'FlopCounterMode' provided by PyTorch.
    # 'FlopCounterMode' provides an analytical flop counter that calculates (i.e. does not actually run an operation and count)
    # for a handpicked list of functions.
    # TODO: Find a way to count flops for other operations.
    # Will return 0, False if the flop count cannot be obtained.
    def estimate_flop_count(self, fx_node: fx.Node) -> int:
        success, args, kwargs = fx_utils.get_fake_args_kwargs(fx_node)
        if not success:
            # TODO: Add logger, and make this print only in verbose mode.
            # print(f"{node_debug_id_str(fx_node)} has flop not countable operator: {fx_node.target._opname}") # noqa: ERA001
            return 0, False

        with FlopCounterMode(display=False) as flop_counter_mode:
            if fx_node.target._overloadpacket not in flop_counter_mode.flop_registry:
                # TODO: Make this debugging print clean
                if os.environ["RANK"] == "0":
                    print(
                        f"{node_debug_id_str(fx_node)} has operator out of the registry: {fx_node.target._opname}, {fx_node.target._overloadpacket}"
                    )
                return 0, False
            fx_node.target(*args, **kwargs)
            return flop_counter_mode.get_total_flops(), True

    def estimate_tensor_size(self, fx_node: fx.Node) -> tuple:
        success, args, kwargs = fx_utils.get_fake_args_kwargs(fx_node)
        if not success:
            if os.environ["RANK"] == "0":
                print(
                    f"{node_debug_id_str(fx_node)} has estimated flopcount but no tensor_size: {fx_node.target._opname}"
                )
            return 0, (0, 0), (0, 0)

        # Assumption: The first two arguments are the input tensors.
        a = args[0].size()
        b = args[1].size()

        if (
            fx_node.target._overloadpacket != torch.ops.aten.mm
            and fx_node.target._overloadpacket != torch.ops.aten._scaled_mm
        ):
            a = args[1].size()
            b = args[2].size()
        numbytes_per_element = 4  # FP32
        estimated_tensor_size = (
            2 * numbytes_per_element * (a[0] * a[1] + b[0] * b[1] + a[0] * b[1])
        )
        if os.environ["RANK"] == "0":
            print(
                f"Estimated tensor size, a: {a[0]} {a[1]} b: {b[0]} {b[1]} result {estimated_tensor_size}"
            )

        return estimated_tensor_size, a, b

    # --- [MODIFIED] ---
    # 修改函数签名以接收 a_shape 和 b_shape, 用于查询缓存
    def measure_duration_microsecond(
        self, fx_node: fx.Node, a_shape: torch.Size, b_shape: torch.Size
    ) -> int:
        # 如果启用缓存, 首先检查缓存
        if self.use_cache == 1:
            key = (str(tuple(a_shape)), str(tuple(b_shape)))
            if key in self.duration_cache:
                if os.environ.get("RANK") == "0":
                    print(
                        f"Cache hit for shapes {key}. Using duration {self.duration_cache[key]} us."
                    )
                return self.duration_cache[key]
            else:
                if os.environ.get("RANK") == "0":
                    print(f"Cache miss for shapes {key}. Measuring duration...")

        if self.use_cache == 2:
            return 0

        with unset_fake_temporarily():

            def realify_fake_tensor(arg) -> torch.Tensor:
                if not isinstance(arg, fx.Node):
                    return arg
                fake_tensor: torch._subclasses.fake_tensor.FakeTensor = arg.meta["val"]
                faketensor_size = fake_tensor.size()
                # [NOTE]: Abandon the change below. Errs with TP, DTensor. For now, modify PT code to force lowering, at 'graph_compile.py'
                # torch.compile(dynamic=True) needed to trigger AOT lowering (avoid lazy lowering)
                # size_array = []
                # for size_value in fake_tensor.size():
                #     size_value_resolved = int(size_value)
                #     size_array.append(size_value_resolved)
                # faketensor_size = torch.Size(size_array)
                real_tensor = torch.empty(
                    faketensor_size, dtype=fake_tensor.dtype, device=fake_tensor.device
                )
                real_tensor = real_tensor.as_strided(
                    faketensor_size, fake_tensor.stride()
                )
                return real_tensor

            flat_args = [realify_fake_tensor(arg) for arg in fx_node.args]
            flat_kwargs = fx_node.kwargs
            num_warmup_iters = 3
            num_iters = 30

            compute_function = fx_node.target._overloadpacket
            for _ in range(num_warmup_iters):
                compute_function(*flat_args, **flat_kwargs)

            start_cuda_event = torch.cuda.Event(enable_timing=True)
            end_cuda_event = torch.cuda.Event(enable_timing=True)
            start_cpu_measured = time.time()
            start_cuda_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                compute_function(*flat_args, **flat_kwargs)
            end_cuda_event.record(torch.cuda.current_stream())
            end_cpu_measured = time.time()
            torch.cuda.synchronize()
            cpu_time = (
                end_cpu_measured - start_cpu_measured
            ) * 1_000_000  # Second to microsecond
            if os.environ["RANK"] == "0":
                print(
                    f"For fx node {fx_node.name}, duration measured by CPU is {cpu_time}, duration measured by CUDA Events is {start_cuda_event.elapsed_time(end_cuda_event)}"
                )
            total_duration_cuda_event = (
                start_cuda_event.elapsed_time(end_cuda_event) * 1000
            )  # Millisecond to microsecond
            mean_duration_cuda_event = int(total_duration_cuda_event / num_iters)
        return mean_duration_cuda_event

    # --- [END MODIFIED] ---

    # --- [MODIFIED] ---
    # 移除 csv_reader 参数, 因为缓存现在由 self.duration_cache 管理
    def _profile_comp_node(
        self, fx_node: fx.Node
    ) -> Tuple[int, int, torch.Size, torch.Size, int]:
        estimated_flops, can_get_real_optarg = self.estimate_flop_count(fx_node)

        estimated_tensor_size = 0
        a_shape = torch.Size()
        b_shape = torch.Size()
        estimated_duration = 0

        if can_get_real_optarg:
            estimated_tensor_size, a_shape, b_shape = self.estimate_tensor_size(fx_node)
            # 将 a_shape 和 b_shape 传递给测量函数以使用缓存
            estimated_duration = self.measure_duration_microsecond(
                fx_node, a_shape, b_shape
            )

        return (
            estimated_flops,
            estimated_tensor_size,
            a_shape,
            b_shape,
            estimated_duration,
        )

    # --- [END MODIFIED] ---

    # --- [MODIFIED] ---
    # 重构此函数, 使其接收预先计算好的性能数据, 避免重复调用 _profile_comp_node
    def create_comp_node(
        self, fx_node: fx.Node, flops: int, tensor_size: int, duration: int
    ) -> ChakraNode:
        node_name = fx_node.name
        chakra_node = self.create_chakra_node(node_name, COMP_NODE)

        chakra_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        chakra_node.attr.append(ChakraAttr(name="num_ops", int64_val=flops))
        chakra_node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        chakra_node.duration_micros = duration

        return chakra_node

    # --- [END MODIFIED] ---

    def record_fx_node(self, fx_node: fx.Node):
        if fx_node.name in self.fxnode_name_lookup_map:
            # Sanity check. This is VERY unlikely.
            print(
                f"{node_debug_id_str(fx_node)} Already has a node with the same name!"
            )
        self.fxnode_name_lookup_map[fx_node.name] = fx_node

    def add_to_chakra_graph(self, chakra_node: ChakraNode, fx_node: fx.Node):
        encode_message(self.et_file, chakra_node)
        self.fxname_chakraid_map[fx_node.name] = chakra_node.id

    def add_upstream_dependency(self, chakra_node: ChakraNode, fx_node: fx.Node):
        upstream_search_queue = fx_node.all_input_nodes
        while len(upstream_search_queue) > 0:
            upstream_candidate = upstream_search_queue.pop()
            if (
                upstream_candidate.name == "root"
                or upstream_candidate.op == "placeholder"
            ):
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
        return

    # --- [MODIFIED] ---
    # 更新此函数以处理缓存写入逻辑
    def process_fx_node(self, fx_node: fx.node, comm_setting: dict, csv_writer=None):
        # Record node info in internal lookup tables.
        self.record_fx_node(fx_node)

        # The following conditions check if the FX Node is worth converting to a Chakra Node.
        # Skip nodes that are 1) placeholders 2) insignificant compute
        if (
            fx_node.name == "root"
            or fx_node.op in ["placeholder", "output"]
            or "getitem" in fx_node.name
        ):
            return

        if not isinstance(fx_node.target, OpOverload):
            print(
                f"""{node_debug_id_str(fx_node)},
                After filters, we still have a node whose target is not OpOverload but {type(fx_node.target)}"""
            )
            return

        if fx_node.target._opname in skip_operator_list:
            return

        chakra_node = None
        if is_comm_node(fx_node):
            chakra_node = self.create_comm_node(fx_node, comm_setting)
        elif is_comp_node(fx_node):
            # 对节点进行一次性能分析, 获取所有信息
            flops, tensor_size, a_shape, b_shape, duration = self._profile_comp_node(
                fx_node
            )

            # 使用分析得到的数据创建Chakra节点
            chakra_node = self.create_comp_node(fx_node, flops, tensor_size, duration)

            # 如果是 rank 0, 并且duration有效, 并且csv_writer存在, 则处理缓存写入
            if os.environ.get("RANK") == "0" and duration != 0 and csv_writer:
                a_shape_str = str(tuple(a_shape))
                b_shape_str = str(tuple(b_shape))
                key = (a_shape_str, b_shape_str)

                # 仅在缓存未命中时(即key不在内存缓存中)才写入CSV并更新内存缓存
                if key not in self.duration_cache:
                    csv_writer.writerow(
                        [fx_node.name, a_shape_str, b_shape_str, duration]
                    )
                    # 为了本次运行后续的节点, 更新内存缓存
                    self.duration_cache[key] = duration
        else:
            print(
                f"Node '{fx_node.name}' is neither a communication nor a computation node. Skipping."
            )
            return

        # Add the upstream dependency to the created chakra node.
        if chakra_node:
            self.add_upstream_dependency(chakra_node, fx_node)
            self.add_to_chakra_graph(chakra_node, fx_node)

    # --- [END MODIFIED] ---

    # --- [MODIFIED] ---
    # 更新文件I/O逻辑, 先加载缓存, 然后再以追加模式打开文件写入
    def convert_to_chakra(self, gm: fx.GraphModule):
        # 如果启用缓存, 首先从CSV文件加载现有数据
        if os.environ.get("RANK") == "0":
            timer.mark("D_start")
        if self.use_cache == 1:
            if os.environ.get("RANK") == "0":
                print(f"Loading duration cache from {self.csv_path}")
            self._load_duration_cache()
            if os.environ.get("RANK") == "0":
                print(f"Loaded {len(self.duration_cache)} entries into cache.")

        with open(self.filename, "wb") as et:
            self.et_file = et
            encode_message(et, GlobalMetadata(version="0.0.4"))
            comm_setting = {}

            # 仅在 rank 0 上打开CSV文件进行写入, 避免多进程冲突
            if os.environ.get("RANK") == "0":
                file_previously_existed = os.path.exists(self.csv_path)
                with open(self.csv_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    # 如果文件是新建的, 写入表头
                    if not file_previously_existed:
                        writer.writerow(["node_name", "a_shape", "b_shape", "duration"])

                    # 处理所有节点, 传入 writer 用于写入新数据
                    for fx_node in gm.graph.nodes:
                        self.process_fx_node(fx_node, comm_setting, writer)
            else:
                # 其他 rank 不需要写入CSV, 但仍需处理节点以生成各自的 .et 文件
                for fx_node in gm.graph.nodes:
                    self.process_fx_node(fx_node, comm_setting, csv_writer=None)
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if rank == 0:
                # The list must be correctly sized.
                gathered_comm_settings = [None] * world_size
                dist.gather_object(comm_setting, gathered_comm_settings, dst=0)
                final_comm_setting = {}
                for setting_dict in gathered_comm_settings:
                    if setting_dict:
                        final_comm_setting.update(setting_dict)

                print("--- Final Merged comm_setting on Rank 0 ---")
                print(final_comm_setting)
                output_filename = "comm_setting.json"
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(final_comm_setting, f, indent=4, ensure_ascii=False)
                    print(f"Successfully saved merged settings to {output_filename}")
                except TypeError as e:
                    print(f"Error saving to JSON: {e}")
                    print("The dictionary might contain non-serializable types.")
                except Exception as e:
                    print(f"An unexpected error occurred while saving the file: {e}")
                # ---------------------------------------------

            else:
                dist.gather_object(comm_setting, None, dst=0)
            dist.barrier()

            if os.environ.get("RANK") == "0":
                timer.mark("program_end")
                timer.display_results()
            destroy_process_group()
            exit()

    # --- [END MODIFIED] ---
