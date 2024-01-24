#!/bin/bash
RUN=true

DATASET="wikitext"
DATASET_ARG="wikitext2"
STATE="sliced"

PYTHON_SCRIPT="run_finetuning.py"
MODEL_FAMILY="facebook"
MODEL_NAME="opt-66b"

ARGS_FIN="--hf-token ***REMOVED*** --save-dir finetuned_models_${DATASET} --finetune-dataset $DATASET_ARG --ppl-eval-dataset $DATASET_ARG --finetune-train-nsamples 8192 --finetune-train-seqlen 512 --lora-alpha 32 --lora-dropout 0.05 --lora-target-modules k_proj v_proj q_proj out_proj fc1 fc2"
ARGS_LRG="--eval-steps 128 --distribute-model"

HYPERPARAMS=("--finetune-train-batch-size 2 --lora-r 64" "--finetune-train-batch-size 2 --lora-r 64" "--finetune-train-batch-size 3 --lora-r 128")
SPARSITIES=(0.2 0.25 0.3)
GPU_IDS=("0,1,2,3" "4,5,6,7" "8,9,10,11")

echo "Running model size $MODEL_NAME"
MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

# Run the script on different GPUs in the background
for ID in {0..2}; do
    GPU_ID=${GPU_IDS[$ID]}
    SPARSITY=${SPARSITIES[$ID]}
    HYPERPARAM=${HYPERPARAMS[$ID]}
    MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
    OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
    ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $HYPERPARAM $ARGS_LRG"
    echo "$GPU_ID: $ARGS"
    if $RUN; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
        sleep 5
    fi
done