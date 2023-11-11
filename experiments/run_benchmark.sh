#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_benchmark.py"

MODEL_FAMILY="meta-llama"
MODEL_NAME="Llama-2-7b-hf"

#MODEL_FAMILY="facebook"
#MODEL_NAME="opt-125m"

MODEL="${MODEL_FAMILY}/${MODEL_NAME}"
BASE_ARGS="--model=$MODEL --hf_token=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"

BATCH_SIZE_DEN_MODEL=512
BATCH_SIZE_SLI_MODEL=1024
SPARSITY=0.25

# Add 128 to the batch size for the denominator model to account for the
# additional memory required for the fused model.
for TEST_ID in {0..2}; do

    for GPU_ID in {0..1}; do
        ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE_DEN_MODEL"
        echo $GPU_ID : $PYTHON_SCRIPT $ARGS
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
        
        BATCH_SIZE_DEN_MODEL=$((BATCH_SIZE_DEN_MODEL + 128))
    done

    for GPU_ID in {2..3}; do
        LOAD_MODEL_PATH="/data/sliced_models/${MODEL_NAME}_${SPARSITY}.pt"
        ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE_SLI_MODEL --sparsity=$SPARSITY --load_model_path=$LOAD_MODEL_PATH"
        echo $GPU_ID : $PYTHON_SCRIPT $ARGS
        
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
        
        BATCH_SIZE_SLI_MODEL=$((BATCH_SIZE_SLI_MODEL + 128))
    done

    wait
    echo "Completed test $TEST_ID."
done