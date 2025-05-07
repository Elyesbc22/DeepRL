import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import glob
from typing import Dict, List, Any, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.environment import Environment
from src.env.wrappers import DiscretizedActionWrapper
from dqn import DQNAgent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize trained DQN agents")
    
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory containing trained models")
    parser.add_argument("--vis_dir", type=str, default="./visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to visualize")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second in the output GIF")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def visualize_agent(agent: DQNAgent, env: Environment, vis_path: str, num_episodes: int = 1, fps: int = 30):
    """
    Visualize the agent's behavior in the environment and save as a GIF.
    
    Args:
        agent: Trained DQN agent
        env: Environment to visualize in
        vis_path: Path to save the visualization
        num_episodes: Number of episodes to visualize
        fps: Frames per second in the output GIF
    """
    frames = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Render the environment
            frame = env.render()
            frames.append(frame)
            
            # Select action
            action = agent.select_action(obs, eval_mode=True)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            obs = next_obs
        
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}")
    
    # Save frames as GIF
    imageio.mimsave(vis_path, frames, fps=fps)
    print(f"Visualization saved to {vis_path}")

def visualize_all_environments(args):
    """
    Visualize trained agents for all supported environments.
    
    Args:
        args: Command line arguments
    """
    # Create visualization directory
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get list of environments
    env_list = Environment.get_env_list()
    
    for env_name in env_list:
        print(f"Visualizing agent for environment: {env_name}")
        
        # Determine if environment has continuous action space
        temp_env = Environment(env_name)
        is_continuous = temp_env.is_continuous
        temp_env.close()
        
        # Create environment with rendering enabled
        if is_continuous:
            env = DiscretizedActionWrapper(env_name, render_mode="rgb_array", seed=args.seed)
        else:
            env = Environment(env_name, render_mode="rgb_array", seed=args.seed)
        
        # Find the latest model for this environment
        model_pattern = os.path.join(args.model_dir, f"dqn_{env_name}_*.pt")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            print(f"No trained model found for {env_name}, skipping...")
            env.close()
            continue
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]
        print(f"Using model: {model_path}")
        
        # Create agent and load model
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim
        )
        agent.load(model_path)
        
        # Visualize agent
        vis_path = os.path.join(args.vis_dir, f"dqn_{env_name}_visualization.gif")
        visualize_agent(agent, env, vis_path, args.num_episodes, args.fps)
        
        # Close environment
        env.close()

if __name__ == "__main__":
    args = parse_args()
    visualize_all_environments(args)