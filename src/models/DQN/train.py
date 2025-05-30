import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

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
    parser.add_argument("--eval_episodes", type=int, default=5, 
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
    parser.add_argument("--save_best", action="store_true", 
                        help="Save the best model during training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model to resume training from")
    parser.add_argument("--checkpoint_freq", type=int, default=10000,
                        help="Frequency of saving checkpoints (in timesteps)")
    parser.add_argument("--record_video", action="store_true",
                        help="Record a video of the agent playing in the environment")
    parser.add_argument("--video_episodes", type=int, default=1,
                        help="Number of episodes to record in the video")
    parser.add_argument("--video_fps", type=int, default=30,
                        help="Frames per second for the recorded video")

    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_agent(agent: DQNAgent, env: Environment, num_episodes: int = 5) -> float:
    """
    Evaluate the agent on the environment.

    Args:
        agent: DQN agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate for (default: 5)

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

def record_agent_video(agent: DQNAgent, env: Environment, video_path: str, num_episodes: int = 1, fps: int = 30):
    """
    Record a video of the agent playing in the environment.

    Args:
        agent: DQN agent to record
        env: Environment to play in
        video_path: Path to save the video
        num_episodes: Number of episodes to record
        fps: Frames per second for the video
    """
    try:
        import imageio
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        frames = []
        rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                # Render the environment
                frame = env.render()

                # Add text with episode, step, and reward information
                if isinstance(frame, np.ndarray):
                    # Convert numpy array to PIL Image
                    frame = Image.fromarray(frame)

                # Add text with episode and reward information
                draw = ImageDraw.Draw(frame)
                text = f"Episode: {episode+1}/{num_episodes}, Step: {step}, Reward: {episode_reward:.2f}"
                draw.text((10, 10), text, fill=(255, 255, 255))

                # Convert back to numpy array
                frame = np.array(frame)

                # Add frame to list
                frames.append(frame)

                # Take action
                action = agent.select_action(obs, eval_mode=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                obs = next_obs
                step += 1

            rewards.append(episode_reward)
            print(f"Recorded episode {episode+1}/{num_episodes} with reward {episode_reward:.2f}")

        # Save video
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Video saved to {video_path}")
        print(f"Average reward over {num_episodes} episodes: {np.mean(rewards):.2f}")

        return video_path
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install imageio and Pillow to record videos: pip install imageio Pillow")
        return None

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
    Train a DQN agent with enhanced logging and visualization.

    Args:
        args: Command line arguments

    Returns:
        Dictionary containing training metrics
    """
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Create a specific log directory for this run
    run_id = f"{args.env.replace('-', '_')}_seed{args.seed}"
    log_dir = os.path.join(args.log_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)

    # Clean old checkpoints if not resuming
    if args.resume is None:
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
    print('    ▶ Initializing DQN…')
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

    # Resume training from a saved model if specified
    start_timestep = 0
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"    ▶ Resuming training from {args.resume}")
            checkpoint = torch.load(args.resume)
            agent.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'replay_buffer' in checkpoint:
                agent.replay_buffer = checkpoint['replay_buffer']
            if 'epsilon' in checkpoint:
                agent.epsilon = checkpoint['epsilon']
            if 'timestep' in checkpoint:
                start_timestep = checkpoint['timestep']
                print(f"    ▶ Resuming from timestep {start_timestep}")
        else:
            print(f"Warning: Resume checkpoint {args.resume} not found. Starting from scratch.")

    # Variables to track the best model
    best_eval_reward = float('-inf')
    best_model_path = None
    # Training loop
    print(f'    ▶ Training for {args.total_timesteps} steps…')
    t0 = time.time()

    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    episode_num = 0
    timestep = 0

    rewards = []
    episode_timesteps = []  # Track the timestep at which each episode ends
    eval_rewards = []
    eval_timesteps = []
    losses = []
    epsilons = []

    progress_bar = tqdm(total=args.total_timesteps, desc="Training", position=2, leave=False)

    # Track time for ETA calculation
    start_time = time.time()
    last_print_time = start_time
    update_interval = 10  # seconds between ETA updates

    # Start from the appropriate timestep if resuming
    for t in range(start_timestep, args.total_timesteps):
        timestep = t + 1  # Start from 1 for better readability

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
            episode_timesteps.append(timestep)  # Track the timestep at which this episode ended

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

        # Save checkpoint at specified frequency
        if args.checkpoint_freq > 0 and t % args.checkpoint_freq == 0 and t > 0:
            checkpoint_path = os.path.join(args.save_dir, f"dqn_{args.env}_checkpoint_{t}.pt")
            checkpoint = {
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'timestep': t,
                'env': args.env,
                'seed': args.seed
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"    ▶ Checkpoint saved at step {t}: {checkpoint_path}")

        # Evaluate agent
        if t % args.eval_freq == 0:
            eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
            eval_rewards.append(eval_reward)
            eval_timesteps.append(timestep)

            # Log evaluation results
            print(f"    ▶ Step {timestep}: Eval mean={eval_reward:.2f}")

            # Save agent
            model_path = os.path.join(args.save_dir, f"dqn_{args.env}_{t}.pt")
            agent.save(model_path)

            # Check if this is the best model so far
            if args.save_best and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_model_path = os.path.join(args.save_dir, f"dqn_{args.env}_best.pt")

                # Save best model with additional information
                checkpoint = {
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'timestep': t,
                    'eval_reward': best_eval_reward,
                    'env': args.env,
                    'seed': args.seed
                }
                torch.save(checkpoint, best_model_path)
                print(f"    ✓ New best model with reward {best_eval_reward:.2f} saved to {best_model_path}")

        # Update progress bar with ETA
        progress_bar.update(1)

        # Calculate and display ETA periodically
        current_time = time.time()
        if current_time - last_print_time > update_interval:
            elapsed = current_time - start_time
            progress = (t - start_timestep + 1) / (args.total_timesteps - start_timestep)
            if progress > 0:
                eta = elapsed / progress - elapsed
                hours, remainder = divmod(eta, 3600)
                minutes, seconds = divmod(remainder, 60)

                # Update progress bar postfix with ETA
                progress_bar.set_postfix({
                    "episode": episode_num,
                    "reward": episode_reward,
                    "epsilon": agent.epsilon,
                    "ETA": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                })

                last_print_time = current_time

    # Final evaluation
    eval_reward = evaluate_agent(agent, eval_env, args.eval_episodes)
    eval_rewards.append(eval_reward)
    eval_timesteps.append(timestep)

    # Check if final model is the best
    is_best_final = False
    if args.save_best and eval_reward > best_eval_reward:
        best_eval_reward = eval_reward
        best_model_path = os.path.join(args.save_dir, f"dqn_{args.env}_best.pt")
        is_best_final = True

        # Save as best model
        checkpoint = {
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'timestep': timestep,
            'eval_reward': best_eval_reward,
            'env': args.env,
            'seed': args.seed,
            'is_final': True
        }
        torch.save(checkpoint, best_model_path)
        print(f"    ✓ Final model is new best with reward {best_eval_reward:.2f}")


    # Save final agent with additional information
    final_model_path = os.path.join(args.save_dir, f"dqn_{args.env}_final.pt")
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'timestep': timestep,
        'eval_reward': eval_reward,
        'best_eval_reward': best_eval_reward if args.save_best else None,
        'best_model_path': best_model_path if args.save_best else None,
        'env': args.env,
        'seed': args.seed,
        'is_final': True
    }
    torch.save(checkpoint, final_model_path)
    print(f"    ✓ Final model saved to {final_model_path}")

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
        "episode_timesteps": [int(t) for t in episode_timesteps],  # Timesteps at which episodes end
        "eval_rewards": [float(r) for r in eval_rewards],
        "eval_timesteps": [int(t) for t in eval_timesteps],
        "losses": [float(l) for l in losses],
        "epsilons": [float(e) for e in epsilons],
        "best_eval_reward": float(best_eval_reward) if args.save_best else None,
        "best_model_path": best_model_path if args.save_best else None,
        "final_eval_reward": float(eval_reward),
        "final_model_path": final_model_path,
        "total_episodes": episode_num,
        "total_timesteps": timestep,
        "training_time_seconds": time.time() - start_time,
        "env": args.env,
        "seed": args.seed,
        "hyperparams": {
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_decay_type": args.epsilon_decay_type,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "target_update_freq": args.target_update_freq,
            "reward_shaping": args.reward_shaping,
            "velocity_coefficient": args.velocity_coefficient
        }
    }

    with open(os.path.join(args.log_dir, f"dqn_{args.env}_metrics_seed={args.seed}.json"), "w") as f:
        json.dump(metrics, f)

    # Record videos if enabled
    if args.record_video:
        print("Recording videos of agent behavior...")

        # Create videos directory
        videos_dir = os.path.join(args.log_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        # Record video of final model
        final_video_path = os.path.join(videos_dir, f"dqn_{args.env}_final.mp4")
        record_agent_video(
            agent=agent,
            env=eval_env,
            video_path=final_video_path,
            num_episodes=args.video_episodes,
            fps=args.video_fps
        )

        # Record video of best model if different from final
        if args.save_best and best_model_path and best_model_path != final_model_path:
            print("Recording video of best model...")
            # Load best model
            best_agent = DQNAgent(
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
            best_checkpoint = torch.load(best_model_path)
            best_agent.load_state_dict(best_checkpoint['model_state_dict'])

            best_video_path = os.path.join(videos_dir, f"dqn_{args.env}_best.mp4")
            record_agent_video(
                agent=best_agent,
                env=eval_env,
                video_path=best_video_path,
                num_episodes=args.video_episodes,
                fps=args.video_fps
            )

            # Update metrics with video paths
            metrics["best_video_path"] = best_video_path

        # Update metrics with video paths
        metrics["final_video_path"] = final_video_path

        # Save updated metrics
        with open(os.path.join(args.log_dir, f"dqn_{args.env}_metrics_seed={args.seed}.json"), "w") as f:
            json.dump(metrics, f)

    # Close environments
    env.close()
    eval_env.close()

    print(f'    ✔ Done in {time.time()-t0:.1f}s')

    return metrics

if __name__ == "__main__":
    args = parse_args()

    # Load config if provided
    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)


    metrics_list = []
    for seed in [239, 366, 165]:
        args.seed = seed
        metrics = train(args)
        metrics_list.append(metrics)

    metrics_averaged = {
        # Average lists of metrics
        "eval_rewards": np.mean([m["eval_rewards"] for m in metrics_list], axis=0).tolist(),
        "eval_timesteps": np.mean([m["eval_timesteps"] for m in metrics_list], axis=0).tolist(),
        "losses": np.mean([m["losses"] for m in metrics_list], axis=0).tolist(),
        "epsilons": np.mean([m["epsilons"] for m in metrics_list], axis=0).tolist(),

        # Average scalar metrics
        "final_eval_reward": np.mean([m["final_eval_reward"] for m in metrics_list]),
        "total_episodes": np.mean([m["total_episodes"] for m in metrics_list]),
        "total_timesteps": np.mean([m["total_timesteps"] for m in metrics_list]),
        "training_time_seconds": np.mean([m["training_time_seconds"] for m in metrics_list]),

        # Keep environment name from first run
        "env": metrics_list[0]["env"],

        # List all seeds used
        "seeds": [m["seed"] for m in metrics_list],

        # Keep hyperparameters from first run as they should be the same
        "hyperparams": metrics_list[0]["hyperparams"]
    }

    with open(os.path.join(args.log_dir, f"dqn_{args.env}_metrics_averaged.json"), "w") as f:
        json.dump(metrics_averaged, f)

    # Create plots directory
    plots_dir = os.path.join(args.log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get current timestamp for plot titles
    # Plot results - Main metrics figure
    plt.figure(figsize=(15, 10))

    # Episode rewards with moving average (plotted against timesteps)
    plt.plot(metrics_averaged["eval_timesteps"], metrics_averaged["eval_rewards"], 'o-', markersize=4)
    plt.title(f"Evaluation Rewards - {args.env}")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward over 5 Episodes")
    plt.grid(True, alpha=0.3)

    # Losses
    plt.savefig(os.path.join(plots_dir, f"dqn_{args.env}_results.png"), dpi=300)

    # Create a second figure comparing training and evaluation rewards
    plt.figure(figsize=(12, 6))

    # If we have evaluation data, plot it against corresponding training data
    if len(metrics["eval_timesteps"]) > 0:
        # Plot evaluation rewards
        plt.plot(metrics["eval_timesteps"], metrics["eval_rewards"], 'o-', label="Evaluation Reward", markersize=6)

        # Plot a smoothed version of training rewards
        if len(metrics["rewards"]) > 10:
            # Create a smoothed version of the training rewards
            smoothed_rewards = pd.Series(metrics["rewards"]).rolling(window=5, min_periods=1).mean()
            # Use actual timesteps from episodes
            plt.plot(
                metrics["episode_timesteps"], 
                smoothed_rewards, 
                'r--', alpha=0.7, 
                label="Smoothed Training Reward"
            )

    plt.title(f"Training vs Evaluation Rewards - {args.env}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"dqn_{args.env}_train_vs_eval.png"), dpi=300)

    print(f"Plots saved to {plots_dir}")