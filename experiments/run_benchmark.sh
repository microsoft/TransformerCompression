#!/bin/bash
RUN=true
DISTRIBUTE=true

PYTHON_SCRIPT="run_benchmark.py"

#MODEL_FAMILY="meta-llama"
#MODEL_NAME="Llama-2-7b-hf"

MODEL_FAMILY="facebook"
MODEL_NAME="opt-125m"

MODEL="${MODEL_FAMILY}/${MODEL_NAME}"
BASE_ARGS="--model=$MODEL --hf_token=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"
if $DISTRIBUTE; then
    BASE_ARGS="$BASE_ARGS --distribute_model"
fi

BATCH_SIZE_DEN_MODEL=256
BATCH_SIZE_SLI_MODEL=256
SPARSITY=0.25

if $DISTRIBUTE; then
    for TEST_ID in {0..2}; do
        GPU_ID="0,1"
        ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE_DEN_MODEL"
        echo $GPU_ID : $PYTHON_SCRIPT $ARGS
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
        
        BATCH_SIZE_DEN_MODEL=$((BATCH_SIZE_DEN_MODEL * 2))

        GPU_ID="2,3"
        LOAD_MODEL_PATH="/data/sliced_models/${MODEL_NAME}_${SPARSITY}.pt"
        ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE_SLI_MODEL --sparsity=$SPARSITY --load_model_path=$LOAD_MODEL_PATH"
        echo $GPU_ID : $PYTHON_SCRIPT $ARGS
        
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
        
        BATCH_SIZE_SLI_MODEL=$((BATCH_SIZE_SLI_MODEL * 2))

        wait
        echo "Completed test $TEST_ID."
    done

else
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
fi

echo "Completed all tests."