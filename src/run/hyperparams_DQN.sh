#!/bin/bash
#SBATCH --job-name=hyperparam_DQN_RL
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load any necessary modules
module load gcc python openmpi py-torch 

# Activate virtual environment if needed
source .venv/bin/activate 

# Run the script
python3 src/models/DQN/hyperparam_search.py "$@"
