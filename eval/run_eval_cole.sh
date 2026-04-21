#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COLE_DIR="$SCRIPT_DIR/COLE"

echo "Running COLE Benchmark Evaluation..."
echo "  Project: $PROJECT_DIR"
echo "  COLE:    $COLE_DIR"
echo ""

# -------------------------------------------------------
# 1. Clone COLE if not already present
# -------------------------------------------------------
if [ ! -d "$COLE_DIR" ]; then
    echo "[1/3] Cloning COLE repository..."
    git clone https://github.com/GRAAL-Research/COLE.git "$COLE_DIR"
else
    echo "[1/3] COLE repository already cloned."
fi

# -------------------------------------------------------
# 2. Install dependencies
# -------------------------------------------------------
echo "[2/3] Installing COLE dependencies..."
pip install -q torch transformers accelerate datasets evaluate \
    tqdm scikit-learn scipy pandas numpy python-dotenv aenum \
    python-multipart fastapi peft 2>/dev/null || true

# Optional: bitsandbytes for GPU quantization
pip install -q bitsandbytes 2>/dev/null || echo "  (bitsandbytes not available — will skip 4-bit quantization)"

# -------------------------------------------------------
# 3. Parse arguments and run evaluation
# -------------------------------------------------------
EXTRA_ARGS=""
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            EXTRA_ARGS="$EXTRA_ARGS --max_examples 5"
            shift
            ;;
        --model)
            EXTRA_ARGS="$EXTRA_ARGS --model $2"
            shift 2
            ;;
        --max_examples)
            EXTRA_ARGS="$EXTRA_ARGS --max_examples $2"
            shift 2
            ;;
        --batch_size)
            EXTRA_ARGS="$EXTRA_ARGS --batch_size $2"
            shift 2
            ;;
        --tasks_group)
            EXTRA_ARGS="$EXTRA_ARGS --tasks_group $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "[3/3] Running COLE evaluation..."
if [ "$QUICK" = true ]; then
    echo "  Mode: Quick test (5 examples per task)"
else
    echo "  Mode: Standard evaluation (100 examples per task)"
fi

# If no --model was specified, the Python script defaults to all 6 models:
#   mistral_base, mistral_lora_fl, mistral_lora_wiki,
#   qwen_base,    qwen_lora_fl,    qwen_lora_wiki
echo ""

python3 "$SCRIPT_DIR/run_cole_eval.py" $EXTRA_ARGS

echo ""
echo "Evaluation complete!"
echo "Results saved in: $PROJECT_DIR/cole_benchmark/"
