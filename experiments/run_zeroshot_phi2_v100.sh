#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_zero_shot_tasks.py"
DATASET="wikitext"
TASKS="piqa arc_easy arc_challenge winogrande hellaswag"

MODEL_FAMILY="microsoft"
MODEL_NAMES_SML=("phi-2")

ARGS_SML="--batch-size 64"
LLA_BASE_ARGS="--tasks $TASKS --hf-token ***REMOVED***"

SPARSITIES=(0.2 0.25 0.3 0.5)

GPU_IDS=(0 1 2 3)
for MODEL_NAME in "${MODEL_NAMES_SML[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for ID in {0..3}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="finetuned_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        ARGS="--model=$MODEL --sparsity=$SPARSITY --load-model-path $MODEL_CHKPT $LLA_BASE_ARGS $ARGS_SML"
        echo "$GPU_ID: $ARGS."
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
done