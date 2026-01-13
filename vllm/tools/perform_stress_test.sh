#!/bin/bash

MODEL_NAME="$1"
INPUT_LEN="${2:-1024}"
OUTPUT_LEN="${3:-1024}"


if [ -z "$MODEL_NAME" ]; then
    echo "Error: 请指定模型名称。"
    echo "Usage: $0 <model_name> [input_len] [output_len]"
    exit 1
fi

LOG_FILE="Scaler-stress-${MODEL_NAME}-${INPUT_LEN}-${OUTPUT_LEN}.txt"
LOOP_COUNT=1

# --- 2. 定义通用的 vllm 命令 ---
VLLM_CMD=(
    vllm bench serve
    --model "/llm/models/$MODEL_NAME"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --ignore-eos
    --num-prompt 32
    --trust-remote-code
    --request-rate inf
    --backend vllm
    --host localhost
)

echo "========================================"
echo " 🚀 开始压力测试"
echo " Model : $MODEL_NAME"
echo " I/O   : $INPUT_LEN / $OUTPUT_LEN"
echo " Log   : $LOG_FILE"
echo "========================================"

while true
do
    CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")

    if (( LOOP_COUNT % 20 == 0 )) || (( LOOP_COUNT == 1 )); then
        
        HEADER="=== Loop < $LOOP_COUNT > | Time: $CURRENT_TIME ==="
        
        echo "$HEADER" | tee -a "$LOG_FILE"
        echo "+++ Saving detailed logs to file..." | tee -a "$LOG_FILE"
        
        "${VLLM_CMD[@]}" >> "$LOG_FILE" 2>&1
        
    else
        
        echo "=== Loop < $LOOP_COUNT > | Time: $CURRENT_TIME (Silent Run) ==="
        
        "${VLLM_CMD[@]}"
    fi

    # --- 4. 计数器自增 (现代语法) ---
    (( LOOP_COUNT++ ))
done
