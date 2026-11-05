#!/usr/bin/env bash
# ==============================================================================
# Hema AI - 75k Parameter LLM Pretraining Script
# Supports both External GPU (CUDA) and Safe Local Training (CPU, no MPS)
# ==============================================================================

set -e

# Detect python executable (prefer local venv if present)
if [ -z "$PYTHON" ]; then
    if [ -x "./venv/bin/python3" ]; then
        PYTHON="./venv/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

echo "=== Hema AI: Starting 75k LLM Pretraining ==="
echo "Python binary: $PYTHON"

# Check Python and Torch
$PYTHON -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device Count: {torch.cuda.device_count()}')
else:
    print('Notice: CUDA is not available. Training will safely execute on CPU (MPS disabled).')
"

# Set Stage and Profile variables
export MODEL_PROFILE="${MODEL_PROFILE:-75k}"
export TRAINING_STAGE="${TRAINING_STAGE:-1}"
export DATASET_PATH="${DATASET_PATH:-data/train.csv}"

echo "Configuration:"
echo "  - Profile:        ${MODEL_PROFILE} (~75k parameters)"
echo "  - Stage:          Stage ${TRAINING_STAGE} (Pretraining on all tokens)"
echo "  - Dataset:        ${DATASET_PATH}"
echo "---------------------------------------------------------"

$PYTHON src/train.py

echo "=== Pretraining Complete ==="
echo "Model checkpoints saved to:"
echo "  - models/pretrained.pth"
echo "  - models/rnn_chatbot.pth"
