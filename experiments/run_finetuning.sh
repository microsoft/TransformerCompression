#!/bin/bash
RUN=false

LLA_MODEL_FAMILY="meta-llama"
LLA_MODEL_NAMES=("Llama-2-7b-hf")
# LLA_MODEL_FAMILY="facebook"
# LLA_MODEL_NAMES=("opt-125m")

BASE_ARGS=--hf-token=<ADD YOUR TOKEN HERE>
SLI_MOD_DIR=sliced_models
FIN_MOD_DIR=finetuned_models

FIN_SCRIPT=run_finetuning.py    
FIN_ARGS="--save-dir=$FIN_MOD_DIR"

ZER_SCRIPT="run_zero_shot_tasks.py"
ZER_ARGS="--batch-size=8 --tasks=piqa,arc_easy,arc_challenge"

SPARSITY=0.3

### FINETUNE ###
echo "Running post-slicing finetuning..."
SCRIPT=$FIN_SCRIPT
GPU_ID=2
for MODEL_NAME in "${LLA_MODEL_NAMES[@]}"; do
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    ARGS="--model=$MODEL --load-model-path=${SLI_MOD_DIR}/${MODEL_NAME}_$SPARSITY.pt --sparsity=$SPARSITY $FIN_ARGS $BASE_ARGS"
    echo $GPU_ID : $SCRIPT $ARGS
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
        sleep 5
    fi

    GPU_ID=$((GPU_ID+1))
done

# Wait for all background processes to finish
wait
echo "Done with post-slicing finetuning."

### ZEROSHOT ###
echo "Running zeroshot experiments..."
SCRIPT=$ZER_SCRIPT  
GPU_ID=0
for MODEL_NAME in "${LLA_MODEL_NAMES[@]}"; do
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the un-sliced model
    ARGS="--model=$MODEL $ZER_ARGS $BASE_ARGS"
    echo $GPU_ID : $SCRIPT $ARGS
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
        sleep 5
    fi

    GPU_ID=$((GPU_ID+1))

    # Run the sliced model
    ARGS="--model=$MODEL --load-model-path=${SLI_MOD_DIR}/${MODEL_NAME}_$SPARSITY.pt --sparsity=$SPARSITY $ZER_ARGS $BASE_ARGS "
    echo $GPU_ID : $SCRIPT $ARGS
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
        sleep 5
    fi

    GPU_ID=$((GPU_ID+1))

    # Run the finetuned model
    ARGS="--model=$MODEL --load-model-path=${FIN_MOD_DIR}/${MODEL_NAME}_$SPARSITY.pt --sparsity=$SPARSITY $ZER_ARGS $BASE_ARGS "
    echo $GPU_ID : $SCRIPT $ARGS
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT $ARGS &
        sleep 5
    fi
done

# Wait for all background processes to finish
wait

echo "Done with zeroshot experiments."