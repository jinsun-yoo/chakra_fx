import argparse
import os
import sys

from chakra.schema.protobuf.et_def_pb2 import (
    COMM_COLL_NODE,
    GlobalMetadata,
)
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.src.third_party.utils.protolib import openFileRd as open_file_rd

input_filename = "test"
output_filename = "test"
non_fsdp_pg_list = [0, 9, 10, 11, 12, 13, 14, 15, 16]


def get_pg_name(node: ChakraNode) -> str:
    for val in node.attr:
        if val.name == "pg_name":
            return val.string_val
    return ""


def get_inputs(node: ChakraNode) -> list:
    for val in node.attr:
        if val.name == "inputs":
            return val.string_list.values
    return []


def modify_fsdp_trace(input_filename: str):
    parts = input_filename.rsplit(".", 2)
    if len(parts) != 3 or parts[2] != "et" or not parts[1].isdigit():
        raise ValueError(f"input_filename must be of the form '{{string}}.{{int}}.et', got: {input_filename}")
    prefix, idx, _ = parts
    output_filename = f"{prefix}_ordering.{idx}.et"

    execution_trace = open_file_rd(input_filename)
    node = ChakraNode()
    nodelist = []
    with open(output_filename, "wb") as et:
        global_metadata = GlobalMetadata()
        decode_message(execution_trace, global_metadata)
        encode_message(et, global_metadata)
        while decode_message(execution_trace, node):
            if node.id > 0 and node.type == COMM_COLL_NODE:
                pg_name = get_pg_name(node)
                if int(pg_name) not in non_fsdp_pg_list:
                    # Traverse up the dependency tree until there is a node without any data_deps
                    # Sometimes, an AG may depend on tensor_view, instead of primals (which is the first node w/o any data_dep)
                    # We must traverse up the dependency tree until we reach the true primals node.
                    first_node = node
                    while len(first_node.data_deps) > 0:
                        # if ".0.et" in output_filename:
                        #     print(f"Traversing node id: {first_node.id}, for node {node.id}, data_deps: {first_node.data_deps}")
                        # print(
                        #     f"Will go to new first node id after traversal: {first_node.id} for node {node.id}, data_deps is {first_node.data_deps[0]}"
                        # )
                        next_node_idx = first_node.data_deps[0]
                        first_node = nodelist[next_node_idx]
                        # print(f"New first node id is {first_node.id} for node {node.id} and next_node_idx {next_node_idx}")
                        # for idx, tempnode in enumerate(nodelist):
                        #     print(f"For idx {idx}Temp node id in nodelist: {tempnode.id}")
                        # if node.id == 16:
                        #     exit(1)

                    # Sanity check. 'first_node' should have a 'primals' node as input.
                    sanity_check = False
                    input_nodes = get_inputs(first_node)
                    for input_node in input_nodes:
                        if "primals" in input_node:
                            sanity_check = True

                    if not sanity_check:
                        print(f"Header node {first_node.id} of Node {node.id} does not have input with 'primals'")
                        exit(1)
                    first_node.data_deps.append(first_node.id - 1)  # Force sequential execution of comm

            # Append a snapshot copy of the current node. If we append the same protobuf object
            # repeatedly, every list element will reference the same object and will reflect
            # the most-recently-decoded values instead of the values at append time.
            node_copy = ChakraNode()
            node_copy.CopyFrom(node)
            nodelist.append(node_copy)
            # if ".0.et" in output_filename:
            #     print(f"Append node id {node.id} to nodelist. Total elements: {len(nodelist)}")
        print(f"Starting to write {output_filename} with {len(nodelist)} nodes.")
        for node in nodelist:
            encode_message(et, node)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify FSDP communication nodes in Chakra execution trace.")
    parser.add_argument("--input_dirname", type=str, required=True, help="Specifies the input filename of the Chakra execution trace.")
    args = parser.parse_args()
    if not os.path.isdir(args.input_dirname):
        raise ValueError(f"input_dirname must be a directory, got: {args.input_dirname}")

    for fname in sorted(os.listdir(args.input_dirname)):
        path = os.path.join(args.input_dirname, fname)
        if not os.path.isfile(path) or "ordering" in fname or "bw" in fname:
            continue
        try:
            modify_fsdp_trace(path)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
