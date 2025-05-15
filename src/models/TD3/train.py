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
from src.env.wrappers import RewardVelocityModifierWrapper, ContinuousActionWrapper
from src.env.utils import set_global_seeds
from td3 import TD3Agent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TD3 agent")

    parser.add_argument("--env", type=str, default="Pendulum-v1", 
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1000000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=1000000, 
                        help="Size of replay buffer")
    parser.add_argument("--actor_lr", type=float, default=3e-4, 
                        help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, 
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, 
                        help="Target network update rate")
    parser.add_argument("--policy_noise", type=float, default=0.2, 
                        help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", type=float, default=0.5, 
                        help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", type=int, default=2, 
                        help="Frequency of delayed policy updates")
    parser.add_argument("--exploration_noise", type=float, default=0.1, 
                        help="Standard deviation of exploration noise")
    parser.add_argument("--eval_freq", type=int, default=5000, 
                        help="Frequency of evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--hidden_dim", type=int, default=256, 
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

def evaluate_agent(agent: TD3Agent, env: Environment, num_episodes: int = 10) -> float:
    """
    Evaluate the agent on the environment.

    Args:
        agent: TD3 agent to evaluate
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
        if filename.startswith(f"td3_{env_name}_") and filename.endswith(".pt"):
            file_path = os.path.join(save_dir, filename)
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {filename}")
            except Exception as e:
                print(f"Error removing checkpoint {filename}: {e}")

def train(args: argparse.Namespace):
    """
    Train a TD3 agent.

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
    # TD3 is designed for continuous action spaces
    temp_env = Environment(args.env, seed=args.seed)

    # Check if the environment has a continuous action space
    if not temp_env.is_continuous:
        print(f"Environment {args.env} has a discrete action space. Using ContinuousActionWrapper to adapt it for TD3.")
        temp_env = ContinuousActionWrapper(args.env, seed=args.seed)

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
            # For evaluation environment
            eval_env = RewardVelocityModifierWrapper(
                args.env,
                velocity_coefficient=args.velocity_coefficient,
                seed=args.seed + 100
            )
        else:
            print("Reward shaping is only supported for MountainCar environments. Using base environment.")
            env = temp_env
            # Check if we need to use ContinuousActionWrapper for evaluation environment
            if not Environment(args.env, seed=args.seed + 100).is_continuous:
                eval_env = ContinuousActionWrapper(args.env, seed=args.seed + 100)
            else:
                eval_env = Environment(args.env, seed=args.seed + 100)
    else:
        env = temp_env
        # Check if we need to use ContinuousActionWrapper for evaluation environment
        if not Environment(args.env, seed=args.seed + 100).is_continuous:
            eval_env = ContinuousActionWrapper(args.env, seed=args.seed + 100)
        else:
            eval_env = Environment(args.env, seed=args.seed + 100)

    # Get the maximum action value from the environment
    max_action = float(env.action_space.high[0])

    # Create agent
    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=max_action,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        exploration_noise=args.exploration_noise
    )

    # Training loop
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    episode_num = 0

    rewards = []
    eval_rewards = []
    actor_losses = []
    critic_losses = []

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
        actor_losses.append(update_info["actor_loss"])
        critic_losses.append(update_info["critic_loss"])

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
                "length": episode_length
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
            agent.save(os.path.join(args.save_dir, f"td3_{args.env}_{t}.pt"))
            # Save metrics
            # Convert numpy values to native Python types for JSON serialization
            metrics = {
                "rewards": [float(r) for r in rewards],
                "eval_rewards": [float(r) for r in eval_rewards],
                "actor_losses": [float(l) for l in actor_losses],
                "critic_losses": [float(l) for l in critic_losses]
            }

            with open(os.path.join(args.log_dir, f"td3_{args.env}_metrics.json"), "w") as f:
                json.dump(metrics, f)

        progress_bar.update(1)

    # Final evaluation
    eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
    eval_rewards.append(eval_reward)

    # Save final agent
    agent.save(os.path.join(args.save_dir, f"td3_{args.env}_final.pt"))

    # Save final metrics
    # Convert numpy values to native Python types for JSON serialization
    metrics = {
        "rewards": [float(r) for r in rewards],
        "eval_rewards": [float(r) for r in eval_rewards],
        "actor_losses": [float(l) for l in actor_losses],
        "critic_losses": [float(l) for l in critic_losses]
    }

    with open(os.path.join(args.log_dir, f"td3_{args.env}_metrics.json"), "w") as f:
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
    plt.plot(metrics["actor_losses"])
    plt.title("Actor Losses")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(metrics["critic_losses"])
    plt.title("Critic Losses")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f"td3_{args.env}_results.png"))
