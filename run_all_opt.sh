PYTHON_SCRIPT="run_slicegpt_perplexity.py"
MODEL_FAMILY="facebook"
MODEL_NAMES = {"opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" "opt-13b" "opt-30b"}
SPARSITIES = {0.0 0.2 0.25 0.3}

LOG_DIR = "logs"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running model size $MODEL_NAME"
    MODEL="${MODEL_FAMILY}/${MODEL_NAME}"
    for I in {0..3}; do
        SPARSITY=${SPARSITIES[$I]}
        echo "Running on GPU with sparsity $SPARSITY"
        ARGS="--model=$MODEL --sparsity=$SPARSITY --eval_baseline"
        LOG_FILE="${LOG_DIR}/ppl_gpu_${MODEL_NAME}_${SPARSITY}.log"
        python $PYTHON_SCRIPT $ARGS > $LOG_FILE 2>&1 &
        sleep 1
        done

    # Wait for all background processes to finish
    wait
    done
 
wait
echo "All processes completed."