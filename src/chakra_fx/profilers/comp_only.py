import torch

# Usage: torchrun --nproc-per-node=<number of processes> transformer.py


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(12_288, 6144)
        self.w2 = torch.nn.Linear(6144, 12_288)

    def forward(self, x):
        x_1 = self.w1(x)
        x_2 = self.w2(x_1)
        return x_2


if __name__ == "__main__":
    model = SimpleModel()
    model = model.to("cuda")

    sample_input = torch.rand(1024, 12_288, dtype=torch.float, device="cuda")

    output = model(sample_input)
    torch.cuda.synchronize()

    output.sum().backward()
    torch.cuda.synchronize()
