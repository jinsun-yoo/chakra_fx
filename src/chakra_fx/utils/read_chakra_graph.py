import argparse
import json

from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
)
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from google.protobuf import json_format


def create_chakra_protobuf_from_json(dir_path: str, json_filename: str, num_ranks: int = 8):
    with open(f"{dir_path}/{json_filename}", "r") as json_file:
        json_data = json.load(json_file)
        protobuf_messages = []
        protobuf_messages.append(GlobalMetadata(version="0.0.4"))
        for obj in json_data[1:]:
            chakra_node = ChakraNode()
            json_format.ParseDict(obj, chakra_node)
            protobuf_messages.append(chakra_node)
        for rank in range(num_ranks):
            with open(f"{dir_path}/trace_new.{rank}.et", "wb") as dst_file:
                for message in protobuf_messages:
                    encode_message(dst_file, message)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True)
    parser.add_argument("--json_filename", default="chakra_trace_modified.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_chakra_protobuf_from_json(args.dir_path, args.json_filename)
