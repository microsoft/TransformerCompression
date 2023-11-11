#!/bin/bash
RUN=true

PYTHON_SCRIPT="run_benchmark.py"
HF_TOKEN="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"

MODEL_FAMILIES=("facebook" "meta-llama")
#MODEL_NAMES=("opt-6.7b" "Llama-2-7b-hf" "opt-13b" "Llama-2-13b-hf") # "OPT-66b" "Llama-2-70b-hf")
MODEL_NAMES=("OPT-66b" "Llama-2-70b-hf")
DISTRIBUTE_FLAGS=(true true) # true true)
INITIAL_BATCH_SIZES=(1 1)

for ((i = 0; i < ${#MODEL_NAMES[@]}; i++)); do
    MODEL_FAMILY="${MODEL_FAMILIES[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    DISTRIBUTE=${DISTRIBUTE_FLAGS[$i]}

    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"
    BASE_ARGS="--model=$MODEL --hf_token=$HF_TOKEN"
    if $DISTRIBUTE; then
        BASE_ARGS="$BASE_ARGS --distribute_model"
    fi

    INITIAL_BATCH_SIZE="${INITIAL_BATCH_SIZES[$i]}"
    SPARSITY=0.25

    if $DISTRIBUTE; then
        for TEST_ID in {0..3}; do
            BATCH_SIZE=$INITIAL_BATCH_SIZE
            GPU_ID="0,1"
            ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE"
            echo $GPU_ID : $PYTHON_SCRIPT $ARGS
            if $RUN; then
                CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
                sleep 5
            fi

            GPU_ID="2,3"
            LOAD_MODEL_PATH="/data/sliced_models/${MODEL_NAME}_${SPARSITY}.pt"
            ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE --sparsity=$SPARSITY --load_model_path=$LOAD_MODEL_PATH"
            echo $GPU_ID : $PYTHON_SCRIPT $ARGS
            
            if $RUN; then
                CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
                sleep 5
            fi
            
            INITIAL_BATCH_SIZE=$((INITIAL_BATCH_SIZE * 2))

            wait
            echo "Completed test $TEST_ID."
        done

    else
        for TEST_ID in {0..1}; do
            BATCH_SIZE=$INITIAL_BATCH_SIZE
            for GPU_ID in {0..1}; do
                ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE"
                echo $GPU_ID : $PYTHON_SCRIPT $ARGS
                if $RUN; then
                    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
                    sleep 5
                fi

                BATCH_SIZE=$((INITIAL_BATCH_SIZE * 2))
            done

            BATCH_SIZE=$INITIAL_BATCH_SIZE
            for GPU_ID in {2..3}; do
                LOAD_MODEL_PATH="/data/sliced_models/${MODEL_NAME}_${SPARSITY}.pt"
                ARGS="$BASE_ARGS --batch_size=$BATCH_SIZE --sparsity=$SPARSITY --load_model_path=$LOAD_MODEL_PATH"
                echo $GPU_ID : $PYTHON_SCRIPT $ARGS
                
                if $RUN; then
                    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT $ARGS &
                    sleep 5
                fi
                
                BATCH_SIZE=$((INITIAL_BATCH_SIZE * 2))
            done

            INITIAL_BATCH_SIZE=$((INITIAL_BATCH_SIZE * 4))
            wait
            echo "Completed test $TEST_ID."
        done
    fi

    echo "Completed all tests."
done