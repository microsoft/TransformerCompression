#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_zero_shot_tasks.py"
DATASET="wikitext"
TASKS="piqa arc_easy arc_challenge winogrande hellaswag"

LLA_MODEL_FAMILY="meta-llama"
# LLA_MODEL_NAMES_SML=("Llama-2-7b-hf")
# LLA_MODEL_NAMES_MED=("Llama-2-13b-hf")
LLA_MODEL_NAMES_LRG=("Llama-2-70b-hf")

ARGS_SML="--batch-size 128"
ARGS_MED="--batch-size 16"
ARGS_LRG="--batch-size 16 --distribute-model"
LLA_BASE_ARGS="--tasks $TASKS --hf-token ***REMOVED***"

SPARSITIES=(0.2 0.25 0.3)

GPU_IDS=(0 1 2 3)
for MODEL_NAME in "${LLA_MODEL_NAMES_SML[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

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

GPU_IDS=(4 5 6 7)
for MODEL_NAME in "${LLA_MODEL_NAMES_MED[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for ID in {0..3}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="finetuned_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        ARGS="--model=$MODEL --sparsity=$SPARSITY --load-model-path $MODEL_CHKPT $LLA_BASE_ARGS $ARGS_MED"
        echo "$GPU_ID: $ARGS."
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
done

# GPU_PAIRS=("0,1" "2,3" "4,5")
GPU_PAIRS=("6,7" "8,9" "10,11")
for MODEL_NAME in "${LLA_MODEL_NAMES_LRG[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPU pairs sequentially
    for ID in {0..2}; do
        GPU_ID="${GPU_PAIRS[$ID]}"
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        ARGS="--model=$MODEL --sparsity=$SPARSITY --load-model-path $MODEL_CHKPT $LLA_BASE_ARGS $ARGS_LRG"
        echo "$GPU_ID: $ARGS."
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $LLA_BASE_ARGS $ARGS_LRG &
            sleep 5
        fi
    done
done