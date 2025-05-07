import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import glob
import re
from typing import Dict, List, Any, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.environment import Environment
from src.env.wrappers import DiscretizedActionWrapper
from dqn import DQNAgent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize trained DQN agents")

    parser.add_argument("--model_dir", type=str, default="./saved_models",
                        help="Directory containing trained models")
    parser.add_argument("--vis_dir", type=str, default="./visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to visualize")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second in the output GIF")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_models", type=int, default=5,
                        help="Number of models to visualize for each environment")
    parser.add_argument("--combine", action="store_true",
                        help="Combine all visualizations into a single GIF")
    parser.add_argument("--model_interval", type=int, default=None,
                        help="Interval between models to visualize (e.g., every 10000 episodes)")

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

def combine_gifs(gif_paths: List[str], output_path: str, fps: int = 30):
    """
    Combine multiple GIFs into a single GIF with side-by-side display.

    Args:
        gif_paths: List of paths to GIFs to combine
        output_path: Path to save the combined GIF
        fps: Frames per second in the output GIF
    """
    if not gif_paths:
        print("No GIFs to combine")
        return

    # Load all GIFs
    all_frames = []
    for path in gif_paths:
        try:
            frames = imageio.mimread(path)
            all_frames.append(frames)
        except Exception as e:
            print(f"Error loading GIF {path}: {e}")

    if not all_frames:
        print("No valid GIFs found")
        return

    # Find the minimum number of frames across all GIFs
    min_frames = min(len(frames) for frames in all_frames)

    # Resize all frames to the same size
    frame_height, frame_width = all_frames[0][0].shape[:2]

    # Create combined frames
    combined_frames = []
    for i in range(min_frames):
        # Get the i-th frame from each GIF
        frames_to_combine = [frames[i] for frames in all_frames if i < len(frames)]

        # Create a new frame with all frames side by side
        combined_width = frame_width * len(frames_to_combine)
        combined_frame = np.zeros((frame_height, combined_width, 3), dtype=np.uint8)

        # Place each frame in the combined frame
        for j, frame in enumerate(frames_to_combine):
            combined_frame[:, j*frame_width:(j+1)*frame_width] = frame

        combined_frames.append(combined_frame)

    # Save the combined GIF
    imageio.mimsave(output_path, combined_frames, fps=fps)
    print(f"Combined GIF saved to {output_path}")

def select_models_by_interval(model_files: List[str], num_models: int, interval: Optional[int] = None) -> List[str]:
    """
    Select models from the list based on the specified interval or number.

    Args:
        model_files: List of model file paths
        num_models: Number of models to select
        interval: Interval between models (in terms of episode number)

    Returns:
        List of selected model file paths
    """
    # Extract episode numbers from filenames
    episode_numbers = []
    episode_to_file = {}

    for file in model_files:
        # Skip files with "final" in the name
        if "final" in file:
            continue

        # Extract episode number from filename
        match = re.search(r'_(\d+)\.pt$', file)
        if match:
            episode = int(match.group(1))
            episode_numbers.append(episode)
            episode_to_file[episode] = file

    if not episode_numbers:
        return []

    # Sort episode numbers
    episode_numbers.sort()

    selected_episodes = []

    if interval:
        # Select episodes at regular intervals
        start = episode_numbers[0]
        end = episode_numbers[-1]

        # Create a range of episodes at the specified interval
        for ep in range(start, end + 1, interval):
            # Find the closest episode number
            closest = min(episode_numbers, key=lambda x: abs(x - ep))
            if closest not in selected_episodes:
                selected_episodes.append(closest)

        # Limit to the specified number of models
        selected_episodes = selected_episodes[:num_models]
    else:
        # Select evenly spaced episodes
        if len(episode_numbers) <= num_models:
            selected_episodes = episode_numbers
        else:
            # Calculate indices for evenly spaced selection
            indices = np.linspace(0, len(episode_numbers) - 1, num_models, dtype=int)
            selected_episodes = [episode_numbers[i] for i in indices]

    # Convert episodes back to file paths
    selected_files = [episode_to_file[ep] for ep in selected_episodes]

    return selected_files

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

    # Dictionary to store GIF paths for each environment
    env_gifs = {}

    for env_name in env_list:
        print(f"Visualizing agent for environment: {env_name}")

        # Determine if environment has continuous action space
        temp_env = Environment(env_name)
        is_continuous = temp_env.is_continuous
        temp_env.close()

        # Find all models for this environment
        model_pattern = os.path.join(args.model_dir, f"dqn_{env_name}_*.pt")
        model_files = glob.glob(model_pattern)

        if not model_files:
            print(f"No trained model found for {env_name}, skipping...")
            continue

        # Sort by episode number (extracted from filename)
        model_files.sort(key=lambda x: int(re.search(r'_(\d+)\.pt$', x).group(1)) if re.search(r'_(\d+)\.pt$', x) else float('inf'))

        # Select models based on interval or number
        selected_models = select_models_by_interval(model_files, args.num_models, args.model_interval)

        if not selected_models:
            print(f"No suitable models found for {env_name}, skipping...")
            continue

        print(f"Selected {len(selected_models)} models for {env_name}")

        # List to store GIF paths for this environment
        env_gif_paths = []

        # Create environment with rendering enabled
        if is_continuous:
            env = DiscretizedActionWrapper(env_name, render_mode="rgb_array", seed=args.seed)
        else:
            env = Environment(env_name, render_mode="rgb_array", seed=args.seed)

        # Create agent
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim
        )

        # Visualize each selected model
        for i, model_path in enumerate(selected_models):
            print(f"Visualizing model {i+1}/{len(selected_models)}: {model_path}")

            # Extract episode number from filename
            episode_match = re.search(r'_(\d+)\.pt$', model_path)
            episode_num = episode_match.group(1) if episode_match else "unknown"

            # Load model
            agent.load(model_path)

            # Visualize agent
            vis_path = os.path.join(args.vis_dir, f"dqn_{env_name}_ep{episode_num}_visualization.gif")
            visualize_agent(agent, env, vis_path, args.num_episodes, args.fps)

            env_gif_paths.append(vis_path)

        # Close environment
        env.close()

        # Store GIF paths for this environment
        env_gifs[env_name] = env_gif_paths

        # Create a combined GIF for this environment
        if args.combine and len(env_gif_paths) > 1:
            combined_path = os.path.join(args.vis_dir, f"dqn_{env_name}_combined_visualization.gif")
            combine_gifs(env_gif_paths, combined_path, args.fps)

    # Create a combined GIF for all environments if requested
    if args.combine and len(env_gifs) > 1:
        # Get the latest GIF for each environment
        latest_gifs = [paths[-1] for paths in env_gifs.values() if paths]

        if latest_gifs:
            all_combined_path = os.path.join(args.vis_dir, "dqn_all_environments_combined.gif")
            combine_gifs(latest_gifs, all_combined_path, args.fps)
            print(f"Combined GIF for all environments saved to {all_combined_path}")

if __name__ == "__main__":
    args = parse_args()
    visualize_all_environments(args)
