#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_slicegpt_perplexity.py"

OPT_MODEL_FAMILY="facebook"
OPT_MODEL_NAMES_SML=("opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b")
OPT_MODEL_NAMES_MED=("opt-13b")
OPT_MODEL_NAMES_LRG=("opt-30b" "opt-66b")

LLA_MODEL_FAMILY="meta-llama"
LLA_MODEL_NAMES_SML=("Llama-2-7b-hf")
LLA_MODEL_NAMES_MED=("Llama-2-13b-hf")
LLA_MODEL_NAMES_LRG=("Llama-2-70b-hf")

ARGS_SML="--batch_size=10"
ARGS_MED="--batch_size=5 --distribute_model"
ARGS_LRG="--batch_size=2 --distribute_model"

ZERO_SPARSITY_ARGS="--eval_baseline --eval_fused_model"

OPT_BASE_ARGS="--hf_token=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 --cal_nsamples=512 --save_dir=/data/sliced_models/"
LLA_BASE_ARGS="--hf_token=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 --cal_nsamples=1024 --save_dir=/data/sliced_models/"

SPARSITIES=(0.0 0.2 0.25 0.3)

for MODEL_NAME in "${OPT_MODEL_NAMES_SML[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${OPT_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for GPU_ID in {0..3}; do
        SPARSITY=${SPARSITIES[$GPU_ID]}
        echo "Running on GPU $GPU_ID, sparsity $SPARSITY."
        ARGS="--model=$MODEL --sparsity=$SPARSITY"
        if test $SPARSITY == 0.0; then
            ARGS="$ARGS $ZERO_SPARSITY_ARGS"
            echo $ARGS
        fi
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $OPT_BASE_ARGS $ARGS_SML &
            sleep 5
        fi
    done
    
    # Wait for all background processes to finish
    wait
done

echo "Completed small OPT models. Starting medium OPT models."

GPU_PAIRS=("0,1" "2,3")
for MODEL_NAME in "${OPT_MODEL_NAMES_MED[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${OPT_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPU pairs sequentially
    for ((i = 0; i < ${#SPARSITIES[@]}; i += 2)); do
        for ((j = 0; j < ${#GPU_PAIRS[@]}; j++)); do
            GPU_ID="${GPU_PAIRS[$j]}"
            SPARSITY="${SPARSITIES[$((i + j))]}"
            echo "Running on GPUs $GPU_ID, sparsity $SPARSITY"
            ARGS="--model=$MODEL --sparsity=$SPARSITY"
            if test $SPARSITY == 0.0; then
                ARGS="$ARGS $ZERO_SPARSITY_ARGS"
                echo $ARGS
            fi
            if $RUN; then
                CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $OPT_BASE_ARGS $ARGS_MED &
                sleep 5
            fi
        done
        wait
    done
done

echo "Completed medium OPT models. Starting large OPT models."

GPU_ID="0,1,2,3"
for MODEL_NAME in "${OPT_MODEL_NAMES_LRG[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${OPT_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on all GPUs
    for SPARSITY in "${SPARSITIES[@]}"; do
        echo "Running on GPUs $GPU_ID, sparsity $SPARSITY"
        ARGS="--model=$MODEL --sparsity=$SPARSITY"
        if test $SPARSITY == 0.0; then
            ARGS="$ARGS $ZERO_SPARSITY_ARGS"
            echo $ARGS
        fi
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $OPT_BASE_ARGS $ARGS_LRG &
        fi
        wait
    done
done

echo "Completed large OPT models. Starting LLAMA models."

for MODEL_NAME in "${LLA_MODEL_NAMES_SML[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPUs in the background
    for GPU_ID in {0..3}; do
        SPARSITY=${SPARSITIES[$GPU_ID]}
        echo "Running on GPU $GPU_ID, sparsity $SPARSITY."
        ARGS="--model=$MODEL --sparsity=$SPARSITY"
        if test $SPARSITY == 0.0; then
            ARGS="$ARGS $ZERO_SPARSITY_ARGS"
            echo $ARGS
        fi
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $LLA_BASE_ARGS $ARGS_SML &
            sleep 5
        fi
    done
    
    wait
done

echo "Completed small LLAMA models. Starting medium LLAMA models."

GPU_PAIRS=("0,1" "2,3")
for MODEL_NAME in "${LLA_MODEL_NAMES_MED[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on different GPU pairs sequentially
    for ((i = 0; i < ${#SPARSITIES[@]}; i += 2)); do
        for ((j = 0; j < ${#GPU_PAIRS[@]}; j++)); do
            GPU_ID="${GPU_PAIRS[$j]}"
            SPARSITY="${SPARSITIES[$((i + j))]}"
            echo "Running on GPUs $GPU_ID, sparsity $SPARSITY"
            ARGS="--model=$MODEL --sparsity=$SPARSITY"
            if test $SPARSITY == 0.0; then
                ARGS="$ARGS $ZERO_SPARSITY_ARGS"
                echo $ARGS
            fi
            if $RUN; then
                CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $LLA_BASE_ARGS $ARGS_MED &
                sleep 5
            fi
        done
        wait
    done
done

echo "Completed medium LLAMA models. Starting large LLAMA models."

GPU_ID="0,1,2,3"
for MODEL_NAME in "${LLA_MODEL_NAMES_LRG[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${LLA_MODEL_FAMILY}/${MODEL_NAME}"

    # Run the script on all GPUs
    for SPARSITY in "${SPARSITIES[@]}"; do
        echo "Running on GPUs $GPU_ID, sparsity $SPARSITY"
        ARGS="--model=$MODEL --sparsity=$SPARSITY"
        if test $SPARSITY == 0.0; then
            ARGS="$ARGS $ZERO_SPARSITY_ARGS"
            echo $ARGS
        fi
        if $RUN; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS $LLA_BASE_ARGS $ARGS_LRG &
            wait
        fi
    done
done

echo "All processes completed."