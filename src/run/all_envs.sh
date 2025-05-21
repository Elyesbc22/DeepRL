#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p saved_models

# Generate a unique run ID using timestamp
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "Run ID: ${RUN_ID}"

# Seeds to use
SEEDS=(42 123 456)

# Models to train with their respective timesteps
declare -A MODEL_TIMESTEPS
MODEL_TIMESTEPS=(["DQN"]=75000 ["PPO"]=150000 ["TD3"]=400000)

# Environment
ENV="CartPole-v1"

# Submit each job to Slurm
for MODEL in "${!MODEL_TIMESTEPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # Create model-specific directories for this run
    LOG_DIR="logs/${MODEL}_${RUN_ID}/seed_${SEED}"
    SAVE_DIR="saved_models/${MODEL}_${RUN_ID}/seed_${SEED}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${SAVE_DIR}"
    
    echo "Submitting ${MODEL} job with seed ${SEED}"
    
    # Submit the job to Slurm
    sbatch src/run/run_single.sh "${MODEL}" "${SEED}" "${ENV}" "${MODEL_TIMESTEPS[$MODEL]}" "${LOG_DIR}" "${SAVE_DIR}"
  done
done

echo "All jobs submitted for run ID: ${RUN_ID}"