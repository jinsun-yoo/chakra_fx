import torch as torch
import torch.distributed as dist
from torch.fx.graph_module import GraphModule


def setup():
    dist.init_process_group("nccl")  # Use NCCL for GPU or Gloo for CPU
    torch.cuda.set_device(0)


def compiled_all_gather(tensor, gathered_tensors):
    dist.all_gather(gathered_tensors, tensor)


# Compile function for better performance
def custom_backend(gm: GraphModule, example_inputs):
    print("Custom backend called")
    gm.print_readable()
    exit()


compiled_all_gather = torch.compile(compiled_all_gather, backend=custom_backend)


def all_gather_example():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Each process has a tensor with its rank
    tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Perform all_gather
    compiled_all_gather(tensor, gathered_tensors)

    print(f"Rank {rank}: {gathered_tensors}")


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    setup()
    all_gather_example()
    cleanup()
