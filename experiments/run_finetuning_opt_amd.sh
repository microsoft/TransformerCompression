#!/bin/bash
RUN=false
RUN_1_3=false
RUN_2_7=false
RUN_6_7=false
RUN_13=false
RUN_30=true

DATASET="wikitext"
DATASET_ARG="wikitext2"
STATE="sliced"

PYTHON_SCRIPT="run_finetuning.py"
MODEL_FAMILY="facebook"

HYPERPARAMS="--finetune-train-seqlen 2048 --lora-alpha 418 --lora-dropout 0.05 --lora-r 84 --lora-target-modules k_proj v_proj q_proj out_proj fc1 fc2"
ARGS_FIN="--hf-token ***REMOVED*** --save-dir finetuned_models_${DATASET} --finetune-dataset $DATASET_ARG --ppl-eval-dataset $DATASET_ARG --finetune-train-nsamples 8192 $HYPERPARAMS"
ARGS_SML="--eval-steps 64 --finetune-train-batch-size 3"
ARGS_LRG="--eval-steps 128 --finetune-train-batch-size 1 --distribute-model"

SPARSITIES=(0.2 0.25 0.3)

if $RUN_1_3; then
    GPU_IDS=(0 1 2)
    MODEL_NAME="opt-1.3b"
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for ID in {0..2}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
fi

if $RUN_2_7; then
    GPU_IDS=(3 4 5)
    MODEL_NAME="opt-2.7b"
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

    # un the script on different GPUs in the background
    for ID in {0..2}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
fi

if $RUN_6_7; then
    GPU_IDS=(6 7 8)
    MODEL_NAME="opt-6.7b"
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

    # un the script on different GPUs in the background
    for ID in {0..2}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
fi

if $RUN_13; then
    GPU_IDS=(9 10 11)
    MODEL_NAME="opt-13b"
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

    # un the script on different GPUs in the background
    for ID in {0..2}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
fi


if $RUN_30; then
    GPU_IDS=(12 13 14)
    MODEL_NAME="opt-30b"
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"

    # un the script on different GPUs in the background
    for ID in {0..2}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_${DATASET}/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
fi