#!/bin/bash
#SBATCH --job-name=hyperparam_RL
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Load any necessary modules
module load gcc python openmpi py-torch
# Activate virtual environment if needed
source .venv/bin/activate

# Generate a unique run ID using timestamp
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "Run ID: ${RUN_ID}"

# Create necessary directories
mkdir -p logs
mkdir -p saved_models

# Seeds to use
SEEDS=(42 123 456)

# Models to train
MODELS=("DQN" "PPO" "TD3" "SAC")

# Environment
ENV="CartPole-v1"

# Train each model with each seed
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Create model-specific directories for this run
        LOG_DIR="logs/${MODEL}_${RUN_ID}/seed_${SEED}"
        SAVE_DIR="saved_models/${MODEL}_${RUN_ID}/seed_${SEED}"
        
        mkdir -p "${LOG_DIR}"
        mkdir -p "${SAVE_DIR}"
        
        echo "Training ${MODEL} with seed ${SEED}"
        echo "Log directory: ${LOG_DIR}"
        echo "Save directory: ${SAVE_DIR}"
        
        # Run the training script
        python src/models/${MODEL,,}/best_hyperparameters.py \
            --env "${ENV}" \
            --seed "${SEED}" \
            --log_dir "${LOG_DIR}" \
            --save_dir "${SAVE_DIR}"
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "Successfully trained ${MODEL} with seed ${SEED}"
        else
            echo "Failed to train ${MODEL} with seed ${SEED}"
        fi
    done
done

echo "All training jobs completed for run ID: ${RUN_ID}"