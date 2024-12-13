from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from model_profiler import ModelProfiler
from torch.autograd import Variable
from torchvision import datasets, models, transforms


class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

    def __init__(self, data, sizes, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset():
    """Partitioning MNIST."""
    dataset = datasets.CIFAR10(
        "../data_cifar10",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


class ResNetProfiler(ModelProfiler):
    def __init__(self, fxgraph_handler, use_pytorch_ir, run_custom_backend_all_rank: bool):
        self.name = "Resnet18"

        model = models.resnet18().cuda(dist.get_rank())
        model = torch.nn.parallel.DistributedDataParallel(model)
        sample_input = torch.randn(1, 3, 224, 224, device=f"cuda:{dist.get_rank()}")

        super().__init__(
            fxgraph_handler,
            use_pytorch_ir,
            model,
            sample_input,
            run_custom_backend_all_rank,
        )

    def run_training_session(self):
        torch.cuda.set_device(self.rank)
        super().compile_model()

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        epoch_loss = 0.0
        target = torch.randint(0, 1000, (1,), device="cuda")

        optimizer.zero_grad()
        output = self.model(self.sample_input)
        loss = F.nll_loss(output, target)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        print(f"Rank: {dist.get_rank()}")

    """Fancy training using full dataset"""
    """Not defined in super() yet"""

    def run_training_session_fancy(self):
        torch.cuda.set_device(self.rank)
        train_set, bsz = partition_dataset()
        super().compile_model()

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        num_batches = ceil(len(train_set.dataset) / float(bsz))
        num_epochs = 10
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = (
                    Variable(data.cuda(self.rank)),
                    Variable(target.cuda(self.rank)),
                )

                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
            print(f"Rank {dist.get_rank()}, epoch {epoch}: {epoch_loss / num_batches}")
