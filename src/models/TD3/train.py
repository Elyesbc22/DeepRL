import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from typing import Dict, Any, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.environment import Environment
from env.wrappers import DiscretizedActionWrapper
from env.utils import set_global_seeds
from td3 import TD3Agent

ENV_ALIAS = {
    "MountainCarContinuous": "MountainCarContinuous-v0",
    "MountainCar": "MountainCar-v0",
    "CartPole": "CartPole-v1",
    "Acrobot": "Acrobot-v1",
    "Pendulum": "Pendulum-v1"
}

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
    parser.add_argument("--actor_lr", type=float, default=3e-4, 
                        help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, 
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, 
                        help="Target network update rate")
    parser.add_argument("--policy_noise", type=float, default=0.2, 
                        help="Noise added to target policy")
    parser.add_argument("--noise_clip", type=float, default=0.5, 
                        help="Range to clip noise")
    parser.add_argument("--policy_freq", type=int, default=2, 
                        help="Frequency of delayed policy updates")
    parser.add_argument("--expl_noise", type=float, default=0.1, 
                        help="Exploration noise")
    parser.add_argument("--buffer_size", type=int, default=1000000, 
                        help="Replay buffer size")
    parser.add_argument("--start_timesteps", type=int, default=25000, 
                        help="Random exploration steps before training")
    parser.add_argument("--eval_freq", type=int, default=10000, 
                        help="Frequency of evaluation (in timesteps)")
    parser.add_argument("--eval_episodes", type=int, default=10, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--hidden_dim", type=int, default=256, 
                        help="Dimension of hidden layers")
    parser.add_argument("--max_episode_steps", type=int, default=1000, 
                        help="Maximum steps per episode")
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

def evaluate_agent(agent: TD3Agent, env: Environment, num_episodes: int = 10) -> float:
    """Evaluate the agent on the environment."""
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        
        # Normalize state
        if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
            state = np.clip(state, env.observation_space.low, env.observation_space.high)
        
        state = np.array(state, dtype=np.float32).flatten()
        
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done and episode_steps < 1000:
            action = agent.select_action(state, evaluate=True)
            
            # Check if we need to convert for CartPole
            if env.env_name == 'CartPole-v1' and isinstance(action, np.ndarray):
                # Convert to scalar action for CartPole
                action = int(action[0] > 0)  # Threshold at 0: negative → 0, positive → 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Normalize next state
            if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
                next_state = np.clip(next_state, env.observation_space.low, env.observation_space.high)
            
            next_state = np.array(next_state, dtype=np.float32).flatten()

            episode_reward += reward
            state = next_state
            episode_steps += 1

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

def train(args):
    """
    Train a TD3 agent.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with training metrics
    """
    # Set seeds
    set_global_seeds(args.seed)
    
    # Convert env alias to full name if needed
    args.env = ENV_ALIAS.get(args.env, args.env)

    # Create environments
    train_env = Environment(args.env, seed=args.seed)
    eval_env = Environment(args.env, seed=args.seed + 100)
    
    # For discrete environments, use DiscretizedActionWrapper for TD3
    # (TD3 is designed for continuous action spaces)
    if not train_env.is_continuous:
        print(f"Warning: {args.env} has discrete action space. Using DiscretizedActionWrapper.")
        train_env = DiscretizedActionWrapper(args.env, seed=args.seed)
        eval_env = DiscretizedActionWrapper(args.env, seed=args.seed + 100)

    # Get state and action dimensions
    obs, _ = train_env.reset()
    
    # In the train function, add these debugging lines:
    print(f"Initial state shape: {np.array(obs).shape}")

    state_dim = np.array(obs, dtype=np.float32).flatten().shape[0]

    # For debugging
    print(f"State dimension after flattening: {state_dim}")

    # Check if action space is discrete or continuous
    if hasattr(train_env.action_space, 'shape') and len(train_env.action_space.shape) > 0:
        action_dim = train_env.action_space.shape[0]
        max_action = float(train_env.action_space.high[0])
        print(f"Continuous action space with dimension: {action_dim}")
    else:
        action_dim = 1  # For CartPole and other discrete envs, use 1 for action dimension
        max_action = 1.0
        print(f"Discrete action space with {train_env.action_space.n} actions, using action_dim=1")

    print(f"Environment: {args.env}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    
    # Create a test sample to verify dimensions
    test_state = np.array(obs, dtype=np.float32).flatten()
    test_action = np.random.uniform(-max_action, max_action, size=action_dim)
    print(f"Test state shape: {test_state.shape}")
    print(f"Test action shape: {test_action.shape}")
    print(f"Combined input dim for critic: {test_state.shape[0] + test_action.shape[0]}")
    
    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize agent
# Initialize agent with verified dimensions
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=args.hidden_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        expl_noise=args.expl_noise
    )
        
    # Training metrics
    episode_rewards = []
    avg_rewards = []
    eval_rewards = []
    critic_losses = []
    actor_losses = []
    
    state, _ = train_env.reset()
    
    # Normalize state
    if hasattr(train_env.observation_space, 'low') and hasattr(train_env.observation_space, 'high'):
        state = np.clip(state, train_env.observation_space.low, train_env.observation_space.high)
    
    state = np.array(state, dtype=np.float32).flatten()
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    # Progress bar
    progress_bar = tqdm(total=args.total_timesteps, desc="Training")
    
    # Training loop
    for t in range(args.total_timesteps):
        episode_timesteps += 1
        
        # In the training loop, around line 225:
        if t < args.start_timesteps:
            # Random exploration at the beginning
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Check if we need special handling for CartPole
        if train_env.env_name == 'CartPole-v1' and isinstance(action, np.ndarray):
            # Convert to scalar action for CartPole
            action = int(action[0] > 0)  # Threshold at 0: negative → 0, positive → 1     
               
        # Take step in environment
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        
        # Normalize next state
        if hasattr(train_env.observation_space, 'low') and hasattr(train_env.observation_space, 'high'):
            next_state = np.clip(next_state, train_env.observation_space.low, train_env.observation_space.high)
        
        next_state = np.array(next_state, dtype=np.float32).flatten()
        
        # Store experience in replay buffer
        agent.replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        # Update agent if enough samples
        if t >= args.start_timesteps:
            update_info = agent.update()
            critic_losses.append(update_info["critic_loss"])
            actor_losses.append(update_info["actor_loss"])
        
        # If episode is done
        if done or episode_timesteps >= args.max_episode_steps:
            # Log info
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_rewards.append(avg_reward)
            
            # Reset environment
            state, _ = train_env.reset()
            
            # Normalize state
            if hasattr(train_env.observation_space, 'low') and hasattr(train_env.observation_space, 'high'):
                state = np.clip(state, train_env.observation_space.low, train_env.observation_space.high)
            
            state = np.array(state, dtype=np.float32).flatten()
            
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                "episode": episode_num,
                "reward": episode_rewards[-1],
                "avg_reward": avg_reward
            })
        
        # Evaluate agent
        if (t + 1) % args.eval_freq == 0:
            eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
            eval_rewards.append(eval_reward)
            
            # Save agent
            agent.save(os.path.join(args.save_dir, f"td3_{args.env}_{t+1}.pt"))
            
            # Save metrics
            metrics = {
                "episode_rewards": episode_rewards,
                "avg_rewards": avg_rewards,
                "eval_rewards": eval_rewards,
                "critic_losses": critic_losses,
                "actor_losses": actor_losses,
                "timesteps": list(range(0, t+1, args.eval_freq))
            }
            
            with open(os.path.join(args.log_dir, f"td3_{args.env}_metrics.json"), "w") as f:
                json.dump(metrics, f)
    
    # Final evaluation
    eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
    eval_rewards.append(eval_reward)
    
    # Save final agent
    agent.save(os.path.join(args.save_dir, f"td3_{args.env}_final.pt"))
    
    # Save final metrics
    metrics = {
        "episode_rewards": episode_rewards,
        "avg_rewards": avg_rewards,
        "eval_rewards": eval_rewards,
        "critic_losses": critic_losses,
        "actor_losses": actor_losses,
        "timesteps": list(range(0, args.total_timesteps, args.eval_freq)) + [args.total_timesteps]
    }
    
    with open(os.path.join(args.log_dir, f"td3_{args.env}_metrics.json"), "w") as f:
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
    plt.plot(metrics["episode_rewards"])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(metrics["timesteps"], metrics["eval_rewards"])
    plt.title("Evaluation Rewards")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 3)
    plt.plot(metrics["critic_losses"][:1000], label="Critic Loss")
    plt.title("Critic Loss (first 1000 updates)")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(metrics["actor_losses"][:1000], label="Actor Loss")
    plt.title("Actor Loss (first 1000 updates)")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f"td3_{args.env}_results.png"))