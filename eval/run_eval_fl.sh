#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
RESULTS_DIR="$PROJECT_DIR/fl_benchmark"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

echo "Starting FL Benchmark Evaluation..."
echo "  Project: $PROJECT_DIR"
echo "  Models:  $MODELS_DIR"
echo ""

# Define the models to evaluate
MODELS=(
    "mistral_base"
    "mistral_lora_fl"
    "mistral_lora_wiki"
    "qwen_base"
    "qwen_lora_fl"
    "qwen_lora_wiki"
)

# Parse arguments
QUICK_FLAG=""
if [[ "$1" == "--quick" ]]; then
    echo "[!] QUICK MODE enabled: only 5 examples per model."
    QUICK_FLAG="--quick"
fi

# Iterate over models and run evaluation
for model in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$model"
    
    echo "--------------------------------------------"
    echo "Evaluating model: $model"
    echo "Path: $MODEL_PATH"
    echo "--------------------------------------------"
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "[ERROR] Model directory not found: $MODEL_PATH"
        continue
    fi
    
    python3 "$SCRIPT_DIR/fl_benchmark.py" \
        --model_path "$MODEL_PATH" \
        $QUICK_FLAG
    
    echo ""
done

echo "All evaluations complete!"
echo "Results stored in: $RESULTS_DIR"
