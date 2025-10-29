#!/bin/bash
# Run TRUE ASTactic Training
# Usage: bash run_astactic.sh

set -e

echo "========================================="
echo "TRUE ASTactic Training Script"
echo "========================================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo ""
else
    echo "WARNING: nvidia-smi not found"
fi

# Check Python
echo "Python version:"
python --version
echo ""

# Create directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints_astactic
mkdir -p cache
echo ""

# Configuration
DATA_PATH="${DATA_PATH:-./data/naturalproofs}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

echo "Training Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Check data
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data directory not found: $DATA_PATH"
    exit 1
fi

echo "========================================="
echo "Starting TRUE ASTactic training..."
echo "========================================="

python train_astactic.py \
    --data_path "$DATA_PATH" \
    --hidden_size "$HIDDEN_SIZE" \
    --embed_size 256 \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --max_context_length 256 \
    --gradient_accumulation_steps 4 \
    --num_workers 4 \
    --use_amp \
    --checkpoint_dir ./checkpoints_astactic \
    --cache_dir ./cache \
    --log_interval 10 \
    2>&1 | tee logs/astactic_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
