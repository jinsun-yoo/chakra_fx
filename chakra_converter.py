import torch.fx as fx
import torch.distributed as dist
from torch._ops import OpOverload
from torch.utils.flop_counter import FlopCounterMode
import torch._inductor.fx_utils as fx_utils
from  torch._subclasses.fake_tensor import FakeTensor

from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BARRIER,
    BROADCAST,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    METADATA_NODE,
    REDUCE_SCATTER,
    BoolList,
    BytesList,
    DoubleList,
    Fixed32List,
    Fixed64List,
    FloatList,
    GlobalMetadata,
    Int32List,
    Int64List,
    Sfixed32List,
    Sfixed64List,
    Sint32List,
    Sint64List,
    StringList,
    Uint32List,
    Uint64List,
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

# Map from c10d operator string to Chakra collective enumeration
c10d_chakra_map = {
    "all_reduce": ALL_REDUCE,
    "all_gather": ALL_GATHER,
    "all_to_all": ALL_TO_ALL,
    "reduce_scatter": REDUCE_SCATTER,
    "broadcast": BROADCAST,
}

# Operators that we will skip, because they either do nothing meaningful compute wise, or are simple placeholders.
# for (c10d::)wait_tensor, the dependency in Chakra graph already implies that the communication has to finish. No need to add a 'wait' element. 
# TODO: Async comms?
# TODO: Are these operators _really_ not meaningful compute-wise? Need to check
skip_operator_list = ["wait_tensor", "view", "t", "transpose", "split"]

# Functions to help inspecting FX Nodes. 
def fxnode_seqnr(fx_node: fx.Node):
    """Get the sequence number of the FX Node."""
    if fx_node.name == "root":
        return -1
    if fx_node.name == "output":
        return -2
    if fx_node.op == "placeholder": 
        return fx_node.name
    if 'seq_nr' not in fx_node.meta:
        print(f"Node name {fx_node.name} does not have seq_nr")
        return fx_node.name
    return fx_node.meta['seq_nr']

def node_debug_id_str(fx_node: fx.Node): 
    """Human readable string to help identify each node in debug"""
    return f"Node name:'{fx_node.name}', seq_id:{fxnode_seqnr(fx_node)}"

def is_comm_node(fx_node:fx.Node):
    target: OpOverload = fx_node.target
    return target.namespace == "_c10d_functional"

def is_comp_node(fx_node:fx.Node):
    target: OpOverload = fx_node.target
    return target.namespace == "aten"

class ChakraConverter():
    def __init__(
        self,
        name : str,
        subgraph_idx : int,
        dir_name: str
    ):
        self.name = name
        self.subgraph_idx = subgraph_idx
        subgraphstr = ""
        if self.subgraph_idx > 0:
            subgraphstr = f'_subgraph-idx_{self.subgraph_idx}'
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
        if 'val' not in fx_node.meta or not isinstance(fx_node.meta['val'], FakeTensor):
            print(f"{node_debug_id_str(fx_node)} is c10d, but fake input is not found")
        else:
            # Use FakeTensor, which is included in the FX Graph as a fake input, to determine communication size
            comm_tensor : FakeTensor = fx_node.meta['val']
            tensor_dtype = comm_tensor.element_size()
            tensor_numelements = comm_tensor.numel()
            comm_size = tensor_dtype * tensor_numelements

        chakra_node = self.create_chakra_node(node_name, COMM_COLL_NODE) 
        chakra_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        chakra_node.attr.append(ChakraAttr(name="comm_type", int64_val=c10d_chakra_map[op_name]))
        chakra_node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        return chakra_node
    
    def create_comp_node(self, fx_node: fx.Node) -> ChakraNode:
        node_name = fx_node.name
        op_name = fx_node.target._opname
        success, args, kwargs = fx_utils.get_fake_args_kwargs(fx_node)
        if not success:
            print(f"{node_debug_id_str(fx_node)} has flop not countable operator: {op_name}")
            counted_flops = 1000 #Arbitrary number
        with FlopCounterMode() as flop_counter_mode:
            fx_node.target(*args, **kwargs)
            counted_flops = flop_counter_mode.get_total_flops()

        chakra_node = self.create_chakra_node(node_name, COMP_NODE)
        chakra_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        chakra_node.attr.append(ChakraAttr(name="num_ops", int64_val=counted_flops))
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
    # Starting from corresponding FX node, iterate the FX Graph upwards until we find nodes that have already been converted to Chakra Nodes.
    def add_upstream_dependency(self, chakra_node: ChakraNode, fx_node: fx.Node):
        upstream_search_queue = fx_node.all_input_nodes
        while len(upstream_search_queue) > 0:
            upstream_candidate = upstream_search_queue.pop()
            if upstream_candidate.name == "root" or upstream_candidate.op == "placeholder": 
                continue
            upstream_name = upstream_candidate.name
            # Sanity check. Because of topological ordering, we should have already recorded any upstream within the FX graph.
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


    def convert_to_chakra(self, gm:fx.GraphModule): 
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
                    print(f"{node_debug_id_str(fx_node)}, After previous filters, we still have a node whose node.target is not OpOverload, instead {type(fx_node.target)}")
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

