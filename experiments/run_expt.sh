#!/bin/bash
RUN=false

LLA_MODEL_FAMILY="meta-llama"
LLA_MODEL_NAMES=("Llama-2-7b-hf")

BASE_ARGS=--hf-token=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
SLI_MOD_DIR=sliced_models
FIN_MOD_DIR=finetuned_models

PPL_SCRIPT=run_slicegpt_perplexity.py
PPL_ARGS="--batch-size=4"

FIN_SCRIPT=run_finetuning.py
FIN_ARGS="--batch-size=4 --cal-nsamples=1024 --save-dir=$FIN_MOD_DIR"

ZER_SCRIPT="run_zero_shot_tasks.py"
ZER_ARGS="--batch-size=4"

SPARSITIES=(0.0 0.1 0.2 0.3)

### PERPLEXITY ###
for MODEL_NAME in "${LLA_MODEL_NAMES[@]}"; do
    echo "Running PERPLEXITY for model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for GPU_ID in {0..3}; do
        SPARSITY=${SPARSITIES[$GPU_ID]}

        ARGS="--model=$MODEL $PPL_ARGS $BASE_ARGS"
        if [ $SPARSITY == 0.0 ]; then
            ARGS="$ARGS --ppl-only"
        else
            ARGS="$ARGS --sparsity=$SPARSITY --load-model-path=${SLI_MOD_DIR}/${MODEL_NAME}_$SPARSITY.pt"
        fi
        SCRIPT=$PPL_SCRIPT
        echo $GPU_ID : $SCRIPT $ARGS
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
            sleep 5
        fi
    done

    # Wait for all background processes to finish
    wait
done

### FINETUNE ###
for MODEL_NAME in "${LLA_MODEL_NAMES[@]}"; do
    echo "Running FINETUNE for model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for GPU_ID in {1..3}; do
        SPARSITY=${SPARSITIES[$GPU_ID]}
        ARGS="--model=$MODEL --load-model-path=${SLI_MOD_DIR}/${MODEL_NAME}_$SPARSITY.pt --sparsity=$SPARSITY $FIN_ARGS $BASE_ARGS"
        SCRIPT=$FIN_SCRIPT
        echo $GPU_ID : $SCRIPT $ARGS
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
            sleep 5
        fi
    done

    # Wait for all background processes to finish
    wait
done

### ZEROSHOT ###
for MODEL_NAME in "${LLA_MODEL_NAMES[@]}"; do
    echo "Running ZEROSHOT for model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for GPU_ID in {0..3}; do
        SPARSITY=${SPARSITIES[$GPU_ID]}
        ARGS="--model=$MODEL $ZER_ARGS $BASE_ARGS"
        if [ $SPARSITY == 0.0 ]; then
            ARGS="$ARGS"
        else
            ARGS="$ARGS --load-model-path=${FIN_MOD_DIR}/${MODEL_NAME}_$SPARSITY.pt --sparsity=$SPARSITY "
        fi
        SCRIPT=$ZER_SCRIPT
        echo $GPU_ID : $SCRIPT $ARGS
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
            sleep 5
        fi
    done

    # Wait for all background processes to finish
    wait
done