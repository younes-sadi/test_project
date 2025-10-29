#!/bin/bash
#SBATCH --job-name=true_astactic        # Job name
#SBATCH --output=logs/astactic_%j.out   # Output log
#SBATCH --error=logs/astactic_%j.err    # Error log
#SBATCH --time=72:00:00                 # 72 hours (AST generation is complex)
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                       # More memory for tree processing
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

echo "========================================="
echo "TRUE ASTactic Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate environment
source ~/venv/bin/activate

# Print info
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p logs
mkdir -p checkpoints_astactic
mkdir -p cache

# Training parameters
DATA_PATH="./data/naturalproofs"
HIDDEN_SIZE=512
EMBED_SIZE=256
BATCH_SIZE=16
NUM_EPOCHS=20
LEARNING_RATE=1e-4
MAX_CONTEXT_LENGTH=256
GRADIENT_ACCUM=4

echo "========================================="
echo "Training Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "========================================="

# Run training
echo "Starting TRUE ASTactic training..."
echo ""

python train_astactic.py \
    --data_path $DATA_PATH \
    --hidden_size $HIDDEN_SIZE \
    --embed_size $EMBED_SIZE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_context_length $MAX_CONTEXT_LENGTH \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --num_workers 4 \
    --use_amp \
    --checkpoint_dir ./checkpoints_astactic \
    --cache_dir ./cache \
    --log_interval 10

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo "========================================="
    echo "Training failed with exit code $?"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi

# Deactivate
deactivate
