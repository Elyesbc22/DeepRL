#!/bin/bash
#SBATCH --job-name=RL_training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

# Get arguments from the command line
MODEL=$1
SEED=$2
ENV=$3
TOTAL_TIMESTEPS=$4
LOG_DIR=$5
SAVE_DIR=$6

# Load any necessary modules
module load gcc python openmpi py-torch

# Activate virtual environment if needed
source .venv/bin/activate

echo "Training ${MODEL} with seed ${SEED}"
echo "Environment: ${ENV}"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Log directory: ${LOG_DIR}"
echo "Save directory: ${SAVE_DIR}"

# Skip training if the folder is already filled
if [ -n "$(ls -A "${SAVE_DIR}" 2>/dev/null)" ]; then
  echo "Skipping ${MODEL} with seed ${SEED} as ${SAVE_DIR} is already filled."
  exit 0
fi

# Run the training script
python "src/models/${MODEL}/train.py" \
  --env "${ENV}" \
  --seed "${SEED}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --log_dir "${LOG_DIR}" \
  --save_dir "${SAVE_DIR}"

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "Successfully trained ${MODEL} with seed ${SEED}"
else
  echo "Failed to train ${MODEL} with seed ${SEED}"
  exit 1
fi