import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.backends.common import aot_autograd
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.distributed.tensor.placement_types import Replicate
from torch.fx.graph_module import GraphModule


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
    print("Custom backend called")
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

    gm.print_readable()
    if custom_backend_called:
        exit()
    custom_backend_called = True


def run(rank, world_size, input_size, output_size):
    # Initialize the process group
    dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

    tp_mesh = init_device_mesh("cuda", (world_size,))
    layer_tp_plan = {
        "w1": ColwiseParallel(output_layouts=Replicate()),
    }

    # Create the model
    model = ParallelLinear(input_size, output_size, rank, world_size).cuda()
    model = parallelize_module(model, tp_mesh, layer_tp_plan)

    # Compile the model's forward pass
    model.forward = torch.compile(
        model.forward,
        backend=aot_autograd(fw_compiler=custom_backend),  # , dynamic=True
    )  # dynamic is marked True to enable dynamic shapes and force backward compiling instead of lzy compiling

    # Dummy input
    x = torch.randn(10, input_size, device="cuda", requires_grad=True)

    # Forward pass
    output = model(x)
    if rank == 0:
        print("Output shape:", output.shape)

    # Backward pass
    labels = torch.randint(0, output_size, (10,), device="cuda")
    loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), labels.flatten(0, 1))
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
    input_size = 8
    output_size = 4

    # Run the process
    run(rank, world_size, input_size, output_size)


if __name__ == "__main__":
    main()
