#!/usr/bin/env bash
# ==============================================================================
# Hema AI - 75k Parameter LLM Pretraining Script for External GPU (CUDA)
# ==============================================================================

set -e

echo "=== Hema AI: Starting 75k LLM Pretraining on External GPU ==="

# Check Python and Torch
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device Count: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA is not available. This run will execute on CPU.')
"

# Set Stage and Profile variables
export MODEL_PROFILE="75k"
export TRAINING_STAGE="1"
export DATASET_PATH="data/train.csv"

# Optional environment overrides (can be passed via environment)
# export EPOCHS="50"
# export SAMPLE_LIMIT="0"

echo "Configuration:"
echo "  - Profile:        ${MODEL_PROFILE} (~75k parameters)"
echo "  - Stage:          Stage 1 (Pretraining on all tokens)"
echo "  - Dataset:        ${DATASET_PATH}"
echo "---------------------------------------------------------"

python3 src/train.py

echo "=== Pretraining Complete ==="
echo "Model checkpoints saved to:"
echo "  - models/pretrained.pth"
echo "  - models/rnn_chatbot.pth"
