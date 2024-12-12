import yaml 
import os
from typing import Callable

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed._tensor import DTensor
from chakra_fx.src.chakra_fx.profilers.apply_configuration import apply_configuration
from chakra_fx.src.chakra_fx.profilers.model_profiler import ModelProfiler
from chakra_fx.src.chakra_fx.profilers.nanogpt_model import Block, GPTConfig

from torchtitan.torchtitan.models import model_name_to_cls
from torchtitan.torchtitan.models.llama import llama2_configs, llama3_configs
from torchtitan.torchtitan.models.llama.model import ModelArgs, Transformer
from torchtitan.torchtitan.parallelisms.parallelize_llama import torch_spmd_parallelize
from torchtitan.torchtitan.config_manager import JobConfig
from torchtitan.torchtitan.parallelisms.parallel_dims import ParallelDims
# Usage: torchrun --nproc-per-node=<number of processes> transformer.py

num_iters = 10
batch_size = 8 
sequence_length = 2048
dtype = torch.bfloat16

def create_llama_job_config() -> JobConfig:
    job_config = JobConfig()

    model_config = type('Model', (), {'norm_type': 'layernorm'})
    setattr(job_config, 'model', model_config)

    training_config_dict = {
        'seq_len': sequence_length,
        'mixed_precision_param': 'bfloat16', 
        'mixed_precision_reduce': 'float32',
        'compile': True
    }
    training_config = type('Training', (), training_config_dict)
    setattr(job_config, 'training', training_config)

    float8_config_dict = {
        'enable_float8_linear': False,
    }
    float8_config = type('Float8', (), float8_config_dict)
    setattr(job_config, 'float8', float8_config)

    activation_checkpoint_dict = {
        'mode': 'none',
        'selective_ac_option': 'op'
    }
    activation_checkpoint = type('AC', (), activation_checkpoint_dict)
    setattr(job_config, 'activation_checkpoint', activation_checkpoint)

    experimental_config_dict = {
        'enable_async_tensor_parallel': False,
    }
    experimental_config = type('Experimental', (), experimental_config_dict)
    setattr(job_config, 'experimental', experimental_config)

    return job_config

class LlamaProfiler(ModelProfiler):
    def __init__(
        self,
        fxgraph_handler: Callable[[torch.fx.GraphModule], None],
        use_pytorch_ir: bool,
        run_custom_backend_all_rank: bool,
        dse_config_filepath: str,
        job: str
    ):
        print('start llama profiler')
        self.name = "llama"
        job_config = create_llama_job_config()
        tokenizer_n_words = 12_288

        # Configure parallelDims
        with open(dse_config_filepath, "r") as file:
            data = yaml.safe_load(file)
        dim_parallelizations = data["overall"]["parallelization"]
        dimensions = data["overall"]["dimensions"]

        dp = 1
        tp = 1
        for idx, parallelization in enumerate(dim_parallelizations):
            if parallelization == "tp":
                tp = dimensions[idx]
            if parallelization == "fsdp":
                dp = dimensions[idx]

        dimensions.reverse()
        dim_parallelizations.reverse()

        # Initialize the device mesh
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=tuple(dimensions),
            mesh_dim_names=tuple(dim_parallelizations),
        )
        parallel_dims = ParallelDims(dp=dp, tp=tp, pp=1, world_size=int(os.environ['WORLD_SIZE']), enable_loss_parallel=True, dp_type='fsdp')
        world_mesh = parallel_dims.build_mesh('cuda')

        model_config = llama3_configs["chakrafxmodel"]
        model_config.norm_type = job_config.model.norm_type
        model_config.vocab_size = tokenizer_n_words 
        model_config.max_seq_len = job_config.training.seq_len
        model = Transformer.from_model_args(model_config)
        parallelized_model = {}
        print('created model')

        if dse_config_filepath is not None:
            #parallelized_model = apply_configuration(model, dse_config_filepath)
            if job != "sample":
                job_config.training.compile = False
                print('apply parallelization without compile')
            parallelized_model = torch_spmd_parallelize(model, world_mesh, parallel_dims, job_config)
            device = "cuda"
            parallelized_model.to_empty(device=device)
            parallelized_model.init_weights()
            parallelized_model.train()

        else:
            world_size = int(os.environ["WORLD_SIZE"])
            device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
            tp_model = tp_model.to("cuda")

            # Parallelization plan. Tensor parallel
            tp_model = parallelize_module(
                module=tp_model,
                device_mesh=device_mesh,
                parallelize_plan={
                    "attn.c_attn": ColwiseParallel(),
                    "attn.c_proj": RowwiseParallel(),
                    "mlp.c_fc": ColwiseParallel(),
                    "mlp.c_proj": RowwiseParallel(),
                },
            )
        print('finish applying parallelization')

        sample_input = torch.randint(high= tokenizer_n_words, size = (batch_size, sequence_length), dtype=torch.int64, device="cuda")

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
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1), labels.flatten(0, 1)
        )

    def run_training_session(self):
        #super().compile_model()
        output = self.model(self.sample_input)
        torch.cuda.synchronize()
        labels = torch.rand(batch_size, sequence_length, type=torch.int64, device="cuda")
        loss = self.loss_fn(output, labels)

        del output
        loss.backward()
        torch.cuda.synchronize()
