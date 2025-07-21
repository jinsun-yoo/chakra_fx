import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.backends.common import aot_autograd
from torch.distributed.device_mesh import init_device_mesh
from torch.fx.graph_module import GraphModule
from torchtitan.experiments.simple_fsdp import SimpleFSDPTransformer
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel
from torchtitan.models.llama3.model.args import TransformerModelArgs


# Define a simple linear model
class ParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, rank, world_size):
        super(ParallelLinear, self).__init__()
        self.rank = rank
        self.world_size = world_size

        # Define a Linear layer using DTensor for tensor parallelism
        self.w1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Wrap the input tensor with DTensor
        output = self.w1(x)
        return output


custom_backend_called = False


# Compile function for better performance
def custom_backend(gm: GraphModule, example_inputs):
    global custom_backend_called
    rank = int(os.environ["RANK"])
    if rank != 0:
        if not custom_backend_called:
            custom_backend_called = True
            return
        exit()
    print("Custom backend called for rank", rank)
    gm.print_readable()
    # Check if this is a forward or backward graph
    # You can inspect the graph structure or node names to determine this
    has_backward_ops = any("backward" in str(node) for node in gm.graph.nodes)

    if not custom_backend_called:
        print("Forward pass")
    else:
        print("Backward pass")
        if has_backward_ops:
            print("Detecting backward graph normally")
        else:
            print("Custom backend has been called, but cannot detect backward graph")
    print("Exit custom backend for rank", rank)
    if custom_backend_called:
        exit()
    custom_backend_called = True


def run(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

    # Assume only replicate parallel
    dp_mesh_dim_names = ("dp_replicate",)
    dp_mode = "replicate"
    world_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=dp_mesh_dim_names)

    # Create the model
    tokenizer_n_words = 12_288
    batch_size = 8
    seq_length = 2048
    model_args = TransformerModelArgs(
        dim=256,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        rope_theta=500000,
    )
    model_args.vocab_size = tokenizer_n_words
    model_args.max_seq_len = 2048
    model_args.norm_type = "layernorm"
    model = SimpleFSDPTransformer(model_args).to("cuda:0")
    model = data_parallel(
        model,
        world_mesh[tuple(dp_mesh_dim_names)],
        mode=dp_mode,
        ac_mode="none",
        mp_policy=None,
        tp_mesh=None,
    )

    # Compile the model's forward pass
    torch._inductor.config.reorder_for_peak_memory = False
    model = torch.compile(
        model, backend=aot_autograd(fw_compiler=custom_backend), dynamic=True, fullgraph=True
    )  # dynamic is marked True to enable dynamic shapes and force backward compiling instead of lzy compiling

    # Dummy input
    x = torch.randint(high=tokenizer_n_words, size=(batch_size, seq_length), dtype=torch.int64, device="cuda")

    # Forward pass
    output = model(x)
    if rank == 0:
        print("Output shape:", output.shape)

    # Backward pass
    labels = torch.randint(high=tokenizer_n_words, size=(batch_size, seq_length), dtype=torch.int64, device="cuda")
    loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), labels.flatten(0, 1))
    exit()
    loss.backward()
    torch.cuda.synchronize()
    if rank == 0:
        print("Backward pass completed. x.grad shape:", x.grad.shape)

    # Cleanup
    dist.destroy_process_group()


def main():
    # Get parameters from torchrun environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Run the process
    run(rank, world_size)


if __name__ == "__main__":
    main()
