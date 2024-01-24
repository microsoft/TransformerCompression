#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_finetuning.py"

MODEL_FAMILY="microsoft"
MODEL_NAME="phi-2"

ARGS_FIN="--save-dir finetuned_models_wikitext --finetune-dataset wikitext2 --ppl-eval-dataset wikitext2 --finetune-train-nsamples 8192 --finetune-train-seqlen 1024 --lora-alpha 10 --lora-dropout 0.05 --lora-r 32 --lora-target-modules k_proj v_proj q_proj dense fc1 fc2"
ARGS_SML="--finetune-train-batch-size 3"

SPARSITIES=(0.2 0.25 0.3 0.5)

# Run small models
echo "Running model size $MODEL_NAME"
MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

# Run the script on different GPUs in the background
for GPU_ID in {0..3}; do
    SPARSITY=${SPARSITIES[$GPU_ID]}
    MODEL_CHKPT="sliced_models_wikitext/${MODEL_NAME}_${SPARSITY}.pt"
    OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
    ARGS="--model=$MODEL --sparsity=$SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
    echo "$GPU_ID: $ARGS"
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
        sleep 5
    fi
done