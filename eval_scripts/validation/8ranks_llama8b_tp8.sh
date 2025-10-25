torchrun \
    --nproc-per-node=8 \
    ../../profile_fxgraph.py \
    --exp_tag llama8b_tp8_8ranks \
    --job postexec_chakra \
    --model llama \
    --dse_config_filepath configs/llama_TP8.yml \
    --llama_config 8B
