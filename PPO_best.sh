SEEDS=(2897 6324 5478)
ENVS=("CartPole-v1" "MountainCar-v0" "Acrobot-v1" "MountainCarContinuous-v0")

for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LOG_DIR="logs/best/PPO/$SEED"
        SAVE_DIR="saved_models/best/PPO/$SEED"
        
        # Check if the folder is already filled
        if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR/*$ENV* 2>/dev/null)" ]; then
            echo "Skipping $LOG_DIR as it is already filled."
            continue
        fi
        
        # Create directories if they don't exist
        mkdir -p "$LOG_DIR"
        mkdir -p "$SAVE_DIR"
        
        if [[ "$ENV" == "MountainCar-v0" ]]; then
            TOTAL_TIMESTEPS=1000000
        elif [[ "$ENV" == "MountainCarContinuous-v0" ]]; then
            TOTAL_TIMESTEPS=200000
        else
            TOTAL_TIMESTEPS=100000
        fi

        python src/models/PPO/train.py --log_dir "$LOG_DIR" --save_dir "$SAVE_DIR" --env "$ENV" --learning_rate 0.0003 --gamma 0.999 --gae_lambda 0.9 --clip_ratio 0.1 --value_coef 1.0 --entropy_coef 0.001 --steps_per_update 4096 --epochs_per_update 15 --hidden_dim 128 --batch_size 128 --total_timesteps "$TOTAL_TIMESTEPS" --seed "$SEED"
    done
done

# python src/models/PPO/train.py --log_dir logs/best/PPO/2897 --save_dir saved_models/best/PPO/2897 --env CartPole-v1  --learning_rate 0.0003 --gamma 0.999 --gae_lambda 0.9 --clip_ratio 0.1 --value_coef 1.0 --entropy_coef 0.001 --steps_per_update 4096 --epochs_per_update 15 --hidden_dim 128 --batch_size 128 --total_timesteps 100000 --seed 2897