from logging import config
import os
from typing import List

import torch
import yaml
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torchtitan.config.job_config import ActivationCheckpoint, JobConfig, Training, Model, Float8
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.deepseek_v3.infra.parallelize_deepseek import (
    parallelize_deepseek,
)
from torchtitan.experiments.simple_fsdp.deepseek_v3_model import SimpleFSDPDeepSeekV3Model
from torchtitan.experiments.simple_fsdp.deepseek_v3_parallelize import parallelize_deepseekv3
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torch._dynamo.backends.common import aot_autograd

# from checkpoint import load_weights_from_hf



from src.chakra_fx.profilers.model_profiler import ModelProfiler

num_iters = 10
batch_size = 8
sequence_length = 4096*4
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


class DeepseekProfiler(ModelProfiler):
    def __init__(
        self,
        job: str,
        exp_tag: str,
        fxgraph_actions: List[str],
        run_custom_backend_all_rank: bool,
        use_pytorch_ir: bool = False,
        dse_config_filepath: str = None,
        job_config_filepath: str = None,
        sequential_generation: bool = False
    ):
        print("start deepseek profiler")
        self.name = "deepseek"
        with open(dse_config_filepath, "r") as file:
            data = yaml.safe_load(file)
        if data is None:
            return model

        if sequential_generation:
            print("Start polling")
            self._poll_start(f"{exp_tag}/syncfile.txt")

        with torch.device('meta'):
            model_str = "debugmodel"
            model_args = deepseekv3_configs[model_str]
            model = SimpleFSDPDeepSeekV3Model(model_args)

        dim_parallelizations = data["parallelization"]
        world_size = int(os.environ["WORLD_SIZE"])
        parallel_dims = ParallelDims(
            dp_replicate=dim_parallelizations.get("dp_replicate", 1),
            dp_shard=dim_parallelizations.get("dp_shard", 1),
            tp=dim_parallelizations.get("tp", 1),
            pp=dim_parallelizations.get("pp", 1),
            ep=dim_parallelizations.get("ep", 1),
            cp=dim_parallelizations.get("cp", 1),
            etp=dim_parallelizations.get("etp", 1),
            world_size=world_size,
        )
        tokenizer_n_words = 12_888


        print("Creating & Parallelizing Model")
        job_config = JobConfig(
            training=Training(compile=False, seq_len=sequence_length, mixed_precision_param="float32", mixed_precision_reduce="float32"),
            activation_checkpoint=ActivationCheckpoint(mode="none"),
        )
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        parallelized_model = parallelize_deepseekv3(model, parallel_dims, job_config)

        model_size_bytes = sum(p.numel() * p.element_size() for p in parallelized_model.parameters())
        print(f"Model size: {model_size_bytes / (1024 ** 2):.2f} MB")

        parallelized_model.to_empty(device="cuda:0")
        # with torch.no_grad():
        #     model.init_weights(buffer_device="cuda:0")
        parallelized_model.train()

        print("finish applying parallelization")
        sample_input = torch.randint(high=tokenizer_n_words, size=(batch_size, sequence_length), dtype=torch.int64, device="cuda:0")
        sample_label = torch.randint(high=tokenizer_n_words, size=(batch_size, sequence_length), dtype=torch.int64, device="cuda:0")

        super().__init__(
            parallelized_model,
            sample_input,
            sample_label,
            exp_tag,
            fxgraph_actions,
            run_custom_backend_all_rank,
            use_pytorch_ir,
            sequential_generation=sequential_generation
        )

    def loss_fn(self, pred, labels):
        # TODO(ruisizhang123): temporary fix to enable async TP for full model compile
        if isinstance(pred, DTensor):
            pred._local_tensor = pred._local_tensor.contiguous()
        return torch.nn.functional.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

    def compile_model(self):
        torch._dynamo.config.capture_scalar_outputs=True
        if self.run_custom_backend:
            if self.use_pytorch_ir:
                compiled_model = torch.compile(self.model, backend=self._custom_pytorch_compiler, dynamic=True, fullgraph=True)
            else:
                compiled_model = torch.compile(
                    self.model,
                    backend=aot_autograd(fw_compiler=self._custom_aten_compiler),
                    fullgraph=True,
                )
        else:
            compiled_model = torch.compile(self.model)
        self.model = compiled_model