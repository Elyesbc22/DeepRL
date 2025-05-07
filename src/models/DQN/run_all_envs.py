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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.environment import Environment
from src.env.wrappers import DiscretizedActionWrapper
from src.env.utils import set_global_seeds
from dqn import DQNAgent
from train import train, parse_args, load_config, evaluate_agent

def custom_train(args, env_name, is_continuous=False):
    """
    Custom training function that uses DiscretizedActionWrapper for continuous environments.

    Args:
        args: Command line arguments
        env_name: Name of the environment
        is_continuous: Whether the environment has a continuous action space

    Returns:
        Training metrics
    """
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Set seeds
    set_global_seeds(args.seed)

    # Create environment with appropriate wrapper if needed
    if is_continuous:
        env = DiscretizedActionWrapper(env_name, num_bins=10, seed=args.seed)
        eval_env = DiscretizedActionWrapper(env_name, num_bins=10, seed=args.seed + 100)
    else:
        env = Environment(env_name, seed=args.seed)
        eval_env = Environment(env_name, seed=args.seed + 100)

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

    progress_bar = tqdm(total=args.total_timesteps, desc=f"Training on {env_name}")

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
            agent.save(os.path.join(args.save_dir, f"dqn_{env_name}_{t}.pt"))

            # Save metrics
            metrics = {
                "rewards": rewards,
                "eval_rewards": eval_rewards,
                "losses": losses,
                "epsilons": epsilons
            }

            with open(os.path.join(args.log_dir, f"dqn_{env_name}_metrics.json"), "w") as f:
                json.dump(metrics, f)

        progress_bar.update(1)

    # Final evaluation
    eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
    eval_rewards.append(eval_reward)

    # Save final agent
    agent.save(os.path.join(args.save_dir, f"dqn_{env_name}_final.pt"))

    # Save final metrics
    metrics = {
        "rewards": rewards,
        "eval_rewards": eval_rewards,
        "losses": losses,
        "epsilons": epsilons
    }

    with open(os.path.join(args.log_dir, f"dqn_{env_name}_metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Close environments
    env.close()
    eval_env.close()

    return metrics

def run_all_environments(args):
    """
    Run DQN experiments on all supported environments.

    Args:
        args: Command line arguments
    """
    # Get list of all supported environments
    env_list = Environment.get_env_list()

    # Create directories for combined results
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Dictionary to store metrics for all environments
    all_metrics = {}

    # Run experiments for each environment
    for env_name in env_list:
        print(f"\n{'='*50}")
        print(f"Running DQN on {env_name}")
        print(f"{'='*50}\n")

        # Create a temporary environment to check if it's continuous
        temp_env = Environment(env_name)
        is_continuous = temp_env.is_continuous
        temp_env.close()

        if is_continuous:
            print(f"Warning: {env_name} has a continuous action space. DQN is designed for discrete action spaces.")
            print("Using a discretized version of the action space for this environment.")

        # Train DQN on this environment using custom training function
        metrics = custom_train(args, env_name, is_continuous)

        # Store metrics
        all_metrics[env_name] = metrics

    # Save combined metrics
    with open(os.path.join(args.log_dir, "dqn_all_environments_metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    # Create combined visualization
    create_combined_visualization(all_metrics, args.log_dir)

    return all_metrics

def create_combined_visualization(all_metrics, log_dir):
    """
    Create a combined visualization of results across all environments.

    Args:
        all_metrics: Dictionary containing metrics for all environments
        log_dir: Directory to save the visualization
    """
    # Create a figure for comparing evaluation rewards across environments
    plt.figure(figsize=(15, 10))

    # Plot evaluation rewards for each environment
    for env_name, metrics in all_metrics.items():
        # Get evaluation rewards
        eval_rewards = metrics["eval_rewards"]

        # Create x-axis values (evaluation steps)
        x = np.arange(len(eval_rewards))

        # Plot
        plt.plot(x, eval_rewards, label=env_name)

    plt.title("DQN Performance Across Environments")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Average Evaluation Reward")
    plt.legend()
    plt.grid(True)

    # Save figure
    plt.savefig(os.path.join(log_dir, "dqn_all_environments_comparison.png"))

    # Create a figure with subplots for each environment
    num_envs = len(all_metrics)
    fig, axes = plt.subplots(num_envs, 1, figsize=(15, 5 * num_envs))

    for i, (env_name, metrics) in enumerate(all_metrics.items()):
        ax = axes[i] if num_envs > 1 else axes

        # Plot training rewards
        rewards = metrics["rewards"]
        ax.plot(rewards)
        ax.set_title(f"Training Rewards for {env_name}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "dqn_all_environments_training.png"))

if __name__ == "__main__":
    args = parse_args()

    # Load config if provided
    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    # Run experiments on all environments
    all_metrics = run_all_environments(args)
