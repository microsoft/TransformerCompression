#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_zero_shot_tasks.py"
ARGS_COMMON="--hf_token=***REMOVED*** --tasks=piqa --no_cache"

ARGS_SPECIFIC=("--model=facebook/opt-125m --load_model_path=sliced_models/opt-125m_0.0.pt --sparsity=0.0"
               "--model=facebook/opt-1.3b --load_model_path=sliced_models/opt-1.3b_0.0.pt --sparsity=0.0"
               "--model=facebook/opt-2.7b --load_model_path=sliced_models/opt-2.7b_0.0.pt --sparsity=0.0"
               "--model=facebook/opt-6.7b --load_model_path=sliced_models/opt-6.7b_0.0.pt --sparsity=0.0"
               "--model=meta-llama/Llama-2-7b-hf --load_model_path=sliced_models/Llama-2-7b-hf_0.0.pt --sparsity=0.0 --batch_size=100")

for GPU_ID in {4..4}; do
    ARGS_SPEC=${ARGS_SPECIFIC[GPU_ID]}
    echo $GPU_ID : $ARGS_SPEC $ARGS_COMMON
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID WANDB_MODE="offline" python $PYTHON_SCRIPT $ARGS_COMMON $ARGS_SPEC &
        sleep 5
    fi
done