#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_finetuning.py"
LLA_MODEL_FAMILY="meta-llama"
# LLA_MODEL_NAMES_SML=("Llama-2-7b-hf")
# LLA_MODEL_NAMES_MED=("Llama-2-13b-hf")
LLA_MODEL_NAMES_LRG=("Llama-2-70b-hf")

ARGS_FIN="--hf-token <TOKEN> --save-dir finetuned_models_wikitext --finetune-dataset wikitext2 --ppl-eval-dataset wikitext2 --finetune-train-nsamples 8192 --finetune-train-seqlen 1024 --lora-alpha 10 --lora-dropout 0.05 --lora-r 32 --lora-target-modules k_proj v_proj q_proj o_proj gate_proj up_proj down_proj"
ARGS_SML="--finetune-train-batch-size 3"
ARGS_MED="--finetune-train-batch-size 3"
ARGS_LRG=" --finetune-train-batch-size 1 --distribute-model"

SPARSITIES=(0.2 0.25 0.3 0.5)

GPU_IDS=(0 1 2 3)
# Run small models
for MODEL_NAME in "${LLA_MODEL_NAMES_SML[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for ID in {0..3}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_wikitext/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model $MODEL --sparsity $SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_SML"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
done

GPU_IDS=(4 5 6 7)
# Run medium models
for MODEL_NAME in "${LLA_MODEL_NAMES_MED[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for ID in {0..3}; do
        GPU_ID=${GPU_IDS[$ID]}
        SPARSITY=${SPARSITIES[$ID]}
        MODEL_CHKPT="sliced_models_wikitext/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model=$MODEL --sparsity=$SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_MED"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
done

echo Waiting...
wait

GPU_PAIRS=("0,1,2" "3,4,5" "6,7,8" "9,10,11")
for MODEL_NAME in "${LLA_MODEL_NAMES_LRG[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPU pairs sequentially
    for ID in {0..3}; do
        GPU_ID="${GPU_PAIRS[$ID]}"
        SPARSITY="${SPARSITIES[$ID]}"
        MODEL_CHKPT="sliced_models_wikitext/${MODEL_NAME}_${SPARSITY}.pt"
        OUT_DIR="finetuning_${MODEL_NAME}_${SPARSITY}"
        ARGS="--model=$MODEL --sparsity=$SPARSITY --load-model-path $MODEL_CHKPT --st_checkpoint_dir $OUT_DIR $ARGS_FIN $ARGS_MED"
        echo "$GPU_ID: $ARGS"
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
            sleep 5
        fi
    done
done

echo Waiting...
wait

# echo Running zeroshot
# ./run_zeroshot_llama_amd.sh