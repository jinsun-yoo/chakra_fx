import os
from typing import Callable

import torch
from torch.distributed._tensor import DTensor
from torchtitan.config_manager import ActivationCheckpoint, JobConfig, Training
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.simple_fsdp import SimpleFSDPTransformer
from torchtitan.experiments.simple_fsdp.parallelize import parallelize_llama
from torchtitan.models.llama3.model.args import TransformerModelArgs

from src.chakra_fx.profilers.model_profiler import ModelProfiler

# Usage: torchrun --nproc-per-node=<number of processes> transformer.py

num_iters = 10
batch_size = 8
sequence_length = 2048
dtype = torch.bfloat16


def create_llama_job_config() -> JobConfig:
    job_config = JobConfig()

    model_config = type("Model", (), {"norm_type": "layernorm"})
    job_config.model = model_config

    training_config_dict = {
        "seq_len": sequence_length,
        "mixed_precision_param": "bfloat16",
        "mixed_precision_reduce": "float32",
        "compile": True,
    }
    training_config = type("Training", (), training_config_dict)
    job_config.training = training_config

    float8_config_dict = {
        "enable_float8_linear": False,
    }
    float8_config = type("Float8", (), float8_config_dict)
    job_config.float8 = float8_config

    activation_checkpoint_dict = {"mode": "none", "selective_ac_option": "op"}
    activation_checkpoint = type("AC", (), activation_checkpoint_dict)
    job_config.activation_checkpoint = activation_checkpoint

    experimental_config_dict = {
        "enable_async_tensor_parallel": False,
    }
    experimental_config = type("Experimental", (), experimental_config_dict)
    job_config.experimental = experimental_config

    return job_config


class LlamaProfiler(ModelProfiler):
    def __init__(
        self,
        fxgraph_handler: Callable[[torch.fx.GraphModule], None],
        use_pytorch_ir: bool,
        run_custom_backend_all_rank: bool,
        dse_config_filepath: str,
        job: str,
    ):
        print("start llama profiler")
        self.name = "llama"
        # job_config = create_llama_job_config()
        tokenizer_n_words = 12_288

        print("Creating Model")
        model_config = TransformerModelArgs(
            dim=256,
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            rope_theta=500000,
        )
        model_config.vocab_size = tokenizer_n_words
        model_config.max_seq_len = 2048  # job_config.training.seq_len
        model_config.norm_type = "layernorm"  # job_config.model.norm_type
        model = SimpleFSDPTransformer(model_config).to("cuda:0")

        print("Parallelizing model")
        # # Configure parallelDims
        # with open(dse_config_filepath, "r") as file:
        #     data = yaml.safe_load(file)
        # dim_parallelizations = data["overall"]["parallelization"]
        # dimensions = data["overall"]["dimensions"]

        # dp = 1
        # tp = 1
        # for idx, parallelization in enumerate(dim_parallelizations):
        #     if parallelization == "tp":
        #         tp = dimensions[idx]
        #     if parallelization == "fsdp":
        #         dp = dimensions[idx]

        # dimensions.reverse()
        # dim_parallelizations.reverse()

        # # Initialize the device mesh
        # device_mesh = init_device_mesh(
        #     device_type="cuda",
        #     mesh_shape=tuple(dimensions),
        #     mesh_dim_names=tuple(dim_parallelizations),
        # )
        world_size = int(os.environ["WORLD_SIZE"])
        parallel_dims = ParallelDims(
            dp_replicate=world_size // 2,
            dp_shard=world_size // 2,
            tp=1,
            pp=1,
            ep=1,
            cp=1,
            world_size=world_size,
        )

        job_config = JobConfig(
            training=Training(compile=False, seq_len=sequence_length, mixed_precision_param="float32", mixed_precision_reduce="float32"),
            activation_checkpoint=ActivationCheckpoint(mode="none"),
        )

        # if job != "sample":
        #     job_config.training.compile = False
        #     print("apply parallelization without compile")
        parallelized_model = parallelize_llama(model, parallel_dims, job_config)
        # parallelized_model.init_weights()
        # parallelized_model.train()

        print("finish applying parallelization")

        sample_input = torch.randint(high=tokenizer_n_words, size=(batch_size, sequence_length), dtype=torch.int64, device="cuda:0")

        super().__init__(
            fxgraph_handler,
            use_pytorch_ir,
            parallelized_model,
            sample_input,
            run_custom_backend_all_rank,
        )

    def loss_fn(self, pred, labels):
        # TODO(ruisizhang123): temporary fix to enable async TP for full model compile
        if isinstance(pred, DTensor):
            pred._local_tensor = pred._local_tensor.contiguous()
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

    def run_training_session(self):
        output = self.model(self.sample_input)
        torch.cuda.synchronize()
        labels = torch.rand(batch_size, sequence_length, type=torch.int64, device="cuda")
        loss = self.loss_fn(output, labels)

        del output
        loss.backward()
        torch.cuda.synchronize()
