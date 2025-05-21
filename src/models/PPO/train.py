import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from typing import Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.environment import Environment
from env.utils import set_global_seeds
from ppo import PPOAgent

ENV_ALIAS = {
    "MountainCarContinuous": "MountainCarContinuous-v0",
    "MountainCar":          "MountainCar-v0",
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent")
    
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1000000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--steps_per_update", type=int, default=4096, 
                        help="Number of steps to collect before each update")
    parser.add_argument("--epochs_per_update", type=int, default=10, 
                        help="Number of optimization epochs per update")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, 
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, 
                        help="GAE lambda parameter")
    parser.add_argument("--clip_ratio", type=float, default=0.2, 
                        help="PPO clipping parameter")
    parser.add_argument("--value_coef", type=float, default=0.5, 
                        help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, 
                        help="Entropy bonus coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, 
                        help="Maximum norm of gradients for clipping")
    parser.add_argument("--eval_freq", type=int, default=10000, 
                        help="Frequency of evaluation (in timesteps)")
    parser.add_argument("--eval_episodes", type=int, default=10, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--hidden_dim", type=int, default=64, 
                        help="Dimension of hidden layers")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory for saving logs")
    parser.add_argument("--save_dir", type=str, default="saved_models", 
                        help="Directory for saving models")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_agent(agent: PPOAgent, env: Environment, num_episodes: int = 10) -> float:
    """
    Evaluate the agent on the environment.

    Args:
        agent: PPO agent to evaluate
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
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            obs = next_obs

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

def train(args):
    # Set seeds
    set_global_seeds(args.seed)
    
    args.env = ENV_ALIAS.get(args.env, args.env)

    # Create environment
    train_env = Environment(args.env, seed=args.seed)
    eval_env = Environment(args.env, seed=args.seed + 100)
    
    # Get the correct state dimensions from environment
    obs, _ = train_env.reset()
    state_dim = np.asarray(obs, dtype=np.float32).flatten().shape[0]
    
    if train_env.is_continuous:
        action_dim = train_env.action_space.shape[0]
        continuous = True
    else:
        action_dim = train_env.action_space.n
        continuous = False
    
    # Debug information to verify dimensions
    print(f"Environment: {args.env}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Continuous: {continuous}")

    # Create agent with correct state_dim
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        continuous=continuous
    )
    
    # Training loop
    rewards = []
    eval_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    total_steps = 0
    progress_bar = tqdm(total=args.total_timesteps, desc="Training")

    while total_steps < args.total_timesteps:
        # Collect trajectories
        trajs_info = agent.collect_trajectory(train_env, args.steps_per_update)
        rewards.extend([trajs_info["avg_reward"]] * trajs_info["episodes"])
        
        # Update agent
        update_info = agent.update(args.epochs_per_update, args.batch_size)
        
        policy_losses.append(update_info["policy_loss"])
        value_losses.append(update_info["value_loss"])
        entropies.append(update_info["entropy"])
        
        # Update total steps
        total_steps += args.steps_per_update
        progress_bar.update(args.steps_per_update)
        
        # Log progress
        progress_bar.set_postfix({
            "reward": trajs_info["avg_reward"],
            "policy_loss": update_info["policy_loss"],
            "value_loss": update_info["value_loss"]
        })
        
        # Evaluate agent
        if total_steps % args.eval_freq < args.steps_per_update:
            eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
            eval_rewards.append(eval_reward)
            
            # Save agent
            agent.save(os.path.join(args.save_dir, f"ppo_{args.env}_{total_steps}.pt"))
            
            # Save metrics
            metrics = {
                "rewards": rewards,
                "eval_rewards": eval_rewards,
                "policy_losses": policy_losses,
                "value_losses": value_losses,
                "entropies": entropies
            }
            
            with open(os.path.join(args.log_dir, f"ppo_{args.env}_metrics.json"), "w") as f:
                json.dump(metrics, f)

    # Final evaluation
    eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
    eval_rewards.append(eval_reward)

    # Save final agent
    agent.save(os.path.join(args.save_dir, f"ppo_{args.env}_final.pt"))

    # Save final metrics
    metrics = {
        "rewards": rewards,
        "eval_rewards": eval_rewards,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "entropies": entropies
    }

    with open(os.path.join(args.log_dir, f"ppo_{args.env}_metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Close environments
    train_env.close()
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
    plt.figure(figsize=(15, 10))

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
    plt.plot(metrics["policy_losses"], label="Policy Loss")
    plt.plot(metrics["value_losses"], label="Value Loss")
    plt.title("Losses")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(metrics["entropies"])
    plt.title("Policy Entropy")
    plt.xlabel("Update")
    plt.ylabel("Entropy")

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f"ppo_{args.env}_results.png"))