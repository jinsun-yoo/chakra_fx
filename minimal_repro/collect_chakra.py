import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from torch.profiler import ExecutionTraceObserver, profile

from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3 import Transformer


# Define llama model
tokenizer_n_words = 12_288
batch_size = 8
seq_length = 2048

model_config = llama3_configs["debugmodel"]
print(model_config)
model_config.vocab_size = tokenizer_n_words
model_config.max_seq_len = seq_length
model_config.norm_type = "layernorm"
model = Transformer(model_config)
device = "cuda:0"
model.to(device)
world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
parallelize_module(
    model,
    device_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
            use_local_output=True,
        ),
    },
)
for transformer_block in model.layers.values():
    layer_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    parallelize_module(
        module=transformer_block,
        device_mesh=device_mesh,
        parallelize_plan=layer_plan,
    )
sample_input = torch.randint(high=tokenizer_n_words, size=(batch_size, seq_length), dtype=torch.int64, device=device)



rank = int(os.environ["RANK"])
et = ExecutionTraceObserver()
et.register_callback(f'pytorch_trace.{rank}.json')

def kineto_trace_handler(profiler):
    profiler.export_chrome_trace(f"kineto_trace.{rank}.json")

with profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=kineto_trace_handler,
    execution_trace_observer=et,
    record_shapes=True,
) as prof:
    for _ in range(5):
        output = model(sample_input)
        torch.cuda.synchronize()
        labels = torch.randint(high=tokenizer_n_words, size=(batch_size, seq_length), dtype=torch.int64, device=device)
        loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), labels.flatten(0, 1))
        loss.backward()
        torch.cuda.synchronize()
        prof.step()
    torch.distributed.destroy_process_group()
    et.unregister_callback()