import os
from typing import List

import torch
import yaml
from torch.distributed._tensor import DTensor
from torchtitan.config.job_config import ActivationCheckpoint, Float8Linear, JobConfig, Training
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.simple_fsdp.llama3.model import SimpleFSDPTransformer
from torchtitan.experiments.simple_fsdp.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3 import llama3_args
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.protocols.model_converter import build_model_converters

from src.chakra_fx.profilers.model_profiler import ModelProfiler

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
        job: str,
        exp_tag: str,
        fxgraph_actions: List[str],
        run_custom_backend_all_rank: bool,
        use_pytorch_ir: bool = False,
        dse_config_filepath: str = None,
        job_config_filepath: str = None,
        sequential_generation: bool = False,
        use_real_device: bool = True,
        combine_fx_subgraphs: bool = False,
        llama_config: str = "debugmodel",
        graph_passes: List[str] = None,
    ):
        rank = int(os.environ.get("RANK", -1))
        if rank == 0:
            print("start llama profiler")
        self.name = "llama"
        tokenizer_n_words = 12_288

        if sequential_generation:
            print("Start polling")
            self._poll_start(f"{exp_tag}/syncfile.txt")

        if rank == 0:
            print("Creating Model")
        if llama_config == "debugmodel":
            model_config = TransformerModelArgs(
                dim=128,
                n_layers=2,
                n_heads=8,
                n_kv_heads=8,
                rope_theta=500000,
            )
        else:
            model_config = llama3_args[llama_config]
        # model_config = llama3_configs["8B"]
        model_config.vocab_size = tokenizer_n_words
        model_config.max_seq_len = 2048  # job_config.training.seq_len
        # model_config.norm_type = "layernorm"  # job_config.model.norm_type

        # Even if we intend to run on real GPU, the unsplit model might be too large.
        # Therefore, we need to create it on meta device, parallelize it, and *then* materialize in real device.
        with torch.device("meta"):
            model = SimpleFSDPTransformer(model_config)
        # Force initialize 'torch.cuda' to avoid 'is_initialized' check later within initialize_device_mesh
        # Which will try to iterate through cuda devices, causing error.
        # This used to be done within SimpleFSDPTransformer init, but for some reason, no longer done.
        torch.cuda.get_device_capability()

        if rank == 0:
            print("Parallelizing model")
        job_config = JobConfig(
            training=Training(
                seq_len=sequence_length,
                mixed_precision_param="float32",
                mixed_precision_reduce="float32",
            ),
            activation_checkpoint=ActivationCheckpoint(mode="none"),
        )
        if dse_config_filepath is not None:
            parallelized_model = self.apply_configuration(model, dse_config_filepath, job_config)
        else:
            world_size = int(os.environ["WORLD_SIZE"])
            parallel_dims = ParallelDims(
                dp_replicate=world_size // 2,
                dp_shard=world_size // 2,
                tp=1,
                pp=1,
                ep=1,
                etp=1,
                cp=1,
                world_size=world_size,
            )
            parallelized_model = parallelize_llama(model, parallel_dims, job_config)

        if rank == 0:
            print("finish applying parallelization")
        actual_device = "meta"
        if use_real_device:
            if rank == 0:
                print("use cuda device instead of meta device. This might lead to OOM.")
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            actual_device = f"cuda:{local_rank}"
        parallelized_model.to_empty(device=actual_device)

        sample_input = torch.randint(
            high=tokenizer_n_words,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
            device=actual_device,
        )
        sample_label = torch.randint(
            high=tokenizer_n_words,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
            device=actual_device,
        )

        super().__init__(
            parallelized_model,
            sample_input,
            sample_label,
            exp_tag,
            fxgraph_actions,
            run_custom_backend_all_rank,
            use_pytorch_ir,
            sequential_generation=sequential_generation,
            combine_fx_subgraphs=combine_fx_subgraphs,
            graph_passes=graph_passes,
        )

    def apply_configuration(self, model, dse_config_filepath: str, job_config: JobConfig):
        with open(dse_config_filepath, "r") as file:
            data = yaml.safe_load(file)
        if data is None:
            return model

        dim_parallelizations = data["parallelization"]
        world_size = int(os.environ["WORLD_SIZE"])
        parallel_dims = ParallelDims(
            dp_replicate=dim_parallelizations.get("dp_replicate", 1),
            dp_shard=dim_parallelizations.get("dp_shard", 1),
            tp=dim_parallelizations.get("tp", 1),
            pp=dim_parallelizations.get("pp", 1),
            ep=dim_parallelizations.get("ep", 1),
            etp=dim_parallelizations.get("etp", 1),
            cp=dim_parallelizations.get("cp", 1),
            world_size=world_size,
        )
        enable_fp8 = data.get("float8", {})
        if enable_fp8:
            print("Use FP8")
            job_config.quantize.linear.float8 = Float8Linear(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                filter_fqns=["output"],
            )
            job_config.model.converters = ["float8"]

        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        parallelized_model = parallelize_llama(model, parallel_dims, job_config)
        return parallelized_model

    def loss_fn(self, pred, labels):
        # TODO(ruisizhang123): temporary fix to enable async TP for full model compile
        if isinstance(pred, DTensor):
            pred._local_tensor = pred._local_tensor.contiguous()
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
