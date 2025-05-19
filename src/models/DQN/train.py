import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from typing import Dict, List, Any, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.env.environment import Environment
from src.env.wrappers import DiscretizedActionWrapper, RewardVelocityModifierWrapper
from src.env.utils import set_global_seeds
from dqn import DQNAgent
from best_hyperparams import update_best_hyperparams

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN agent")

    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=100000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=10000, 
                        help="Size of replay buffer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, 
                        help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.01, 
                        help="Final exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, 
                        help="Exploration rate decay factor (for exponential) or decay steps (for linear)")
    parser.add_argument("--epsilon_decay_type", type=str, default="exponential", choices=["exponential", "linear"],
                        help="Type of epsilon decay: exponential or linear")
    parser.add_argument("--target_update_freq", type=int, default=10, 
                        help="Frequency of target network updates")
    parser.add_argument("--eval_freq", type=int, default=1000, 
                        help="Frequency of evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--hidden_dim", type=int, default=128, 
                        help="Dimension of hidden layers")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory for saving logs")
    parser.add_argument("--save_dir", type=str, default="saved_models", 
                        help="Directory for saving models")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file")
    parser.add_argument("--reward_shaping", action="store_true", 
                        help="Enable reward shaping with velocity component")
    parser.add_argument("--velocity_coefficient", type=float, default=2,
                        help="Coefficient for velocity in reward shaping")

    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_agent(agent: DQNAgent, env: Environment, num_episodes: int = 10) -> float:
    """
    Evaluate the agent on the environment.

    Args:
        agent: DQN agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate for

    Returns:
        Average reward over the episodes
    """
    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(obs, eval_mode=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            obs = next_obs

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

def clean_checkpoints(save_dir: str, env_name: str):
    """
    Remove old checkpoints for a specific environment.

    Args:
        save_dir: Directory where checkpoints are saved
        env_name: Name of the environment
    """
    if not os.path.exists(save_dir):
        return

    # Get all files in the save directory
    for filename in os.listdir(save_dir):
        # Check if the file is a checkpoint for the current environment
        # This includes both intermediate checkpoints (dqn_{env_name}_{step}.pt)
        # and the final checkpoint (dqn_{env_name}_final.pt)
        if filename.startswith(f"dqn_{env_name}_") and filename.endswith(".pt"):
            file_path = os.path.join(save_dir, filename)
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {filename}")
            except Exception as e:
                print(f"Error removing checkpoint {filename}: {e}")

def train(args: argparse.Namespace):
    """
    Train a DQN agent.

    Args:
        args: Command line arguments
    """
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Clean old checkpoints
    clean_checkpoints(args.save_dir, args.env)

    # Set seeds
    set_global_seeds(args.seed)

    # Create environment
    # Create a temporary environment to check if it has a continuous action space
    temp_env = Environment(args.env, seed=args.seed)

    # Check if the environment has a continuous action space using the is_continuous property
    if temp_env.is_continuous:
        # Use DiscretizedActionWrapper for continuous environments
        base_env = DiscretizedActionWrapper(args.env, seed=args.seed)
        base_eval_env = DiscretizedActionWrapper(args.env, num_bins=5, seed=args.seed + 100)
    else:
        # Use regular Environment for discrete environments
        base_env = temp_env
        base_eval_env = Environment(args.env, seed=args.seed + 100)

    # Apply reward shaping if enabled
    if args.reward_shaping:
        # For training environment
        if 'MountainCar' in args.env:
            print(f"Applying reward shaping with velocity coefficient: {args.velocity_coefficient}")
            env = RewardVelocityModifierWrapper(
                args.env, 
                velocity_coefficient=args.velocity_coefficient,
                seed=args.seed
            )
            # If using discretized wrapper, we need to maintain the discretized action space
            if temp_env.is_continuous:
                env.action_space = base_env.action_space
                env.is_continuous = base_env.is_continuous
                if hasattr(base_env, 'original_action_space'):
                    env.original_action_space = base_env.original_action_space
                    env._discrete_to_continuous = base_env._discrete_to_continuous

            # For evaluation environment
            eval_env = RewardVelocityModifierWrapper(
                args.env,
                velocity_coefficient=args.velocity_coefficient,
                seed=args.seed + 100
            )
            # If using discretized wrapper, we need to maintain the discretized action space
            if temp_env.is_continuous:
                eval_env.action_space = base_eval_env.action_space
                eval_env.is_continuous = base_eval_env.is_continuous
                if hasattr(base_eval_env, 'original_action_space'):
                    eval_env.original_action_space = base_eval_env.original_action_space
                    eval_env._discrete_to_continuous = base_eval_env._discrete_to_continuous
        else:
            print("Reward shaping is only supported for MountainCar environments. Using base environment.")
            env = base_env
            eval_env = base_eval_env
    else:
        env = base_env
        eval_env = base_eval_env

    # Create agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_type=args.epsilon_decay_type,
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq
    )

    # Training loop
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    episode_num = 0

    rewards = []
    eval_rewards = []
    losses = []
    epsilons = []

    progress_bar = tqdm(total=args.total_timesteps, desc="Training")

    for t in range(args.total_timesteps):
        # Select action
        action = agent.select_action(obs)

        # Take step in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)

        # Update agent
        update_info = agent.update()
        losses.append(update_info["loss"])
        epsilons.append(agent.epsilon)

        # Update current state
        obs = next_obs
        episode_reward += reward
        episode_length += 1

        # If episode is done, reset environment
        if done:
            rewards.append(episode_reward)

            # Log episode information
            progress_bar.set_postfix({
                "episode": episode_num,
                "reward": episode_reward,
                "length": episode_length,
                "epsilon": agent.epsilon
            })

            # Reset environment
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_num += 1

        # Evaluate agent
        if t % args.eval_freq == 0:
            eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
            eval_rewards.append(eval_reward)

            # Save agent
            agent.save(os.path.join(args.save_dir, f"dqn_{args.env}_{t}.pt"))
            # Save metrics
            # Convert numpy values to native Python types for JSON serialization
            metrics = {
                "rewards": [float(r) for r in rewards],
                "eval_rewards": [float(r) for r in eval_rewards],
                "losses": [float(l) for l in losses],
                "epsilons": [float(e) for e in epsilons]
            }

            with open(os.path.join(args.log_dir, f"dqn_{args.env}_metrics.json"), "w") as f:
                json.dump(metrics, f)

        progress_bar.update(1)

    # Final evaluation
    eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
    eval_rewards.append(eval_reward)

    # Save final agent
    agent.save(os.path.join(args.save_dir, f"dqn_{args.env}_final.pt"))

    # Update best hyperparameters if current results are better
    current_hyperparams = {
        "hidden_dim": args.hidden_dim,
        "learning_rate": agent.optimizer.param_groups[0]['lr'],
        "gamma": agent.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": agent.epsilon_end,
        "epsilon_decay": agent.epsilon_decay,
        "buffer_size": agent.replay_buffer.buffer.maxlen,
        "batch_size": agent.batch_size,
        "target_update_freq": agent.target_update_freq,
        "reward_shaping": args.reward_shaping,
        "velocity_coefficient": args.velocity_coefficient
    }

    updated = update_best_hyperparams(args.env, current_hyperparams, eval_reward)
    if updated:
        print(f"Updated best hyperparameters for {args.env} with final evaluation reward: {eval_reward:.2f}")

    # Save final metrics
    # Convert numpy values to native Python types for JSON serialization
    metrics = {
        "rewards": [float(r) for r in rewards],
        "eval_rewards": [float(r) for r in eval_rewards],
        "losses": [float(l) for l in losses],
        "epsilons": [float(e) for e in epsilons]
    }

    with open(os.path.join(args.log_dir, f"dqn_{args.env}_metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Close environments
    env.close()
    eval_env.close()

    return metrics

if __name__ == "__main__":
    args = parse_args()

    # Load config if provided
    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)



    metrics = train(args)

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(metrics["eval_rewards"])
    plt.title("Evaluation Rewards")
    plt.xlabel("Evaluation")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 3)
    plt.plot(metrics["losses"])
    plt.title("Losses")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(metrics["epsilons"])
    plt.title("Exploration Rate")
    plt.xlabel("Update")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f"dqn_{args.env}_results.png"))
