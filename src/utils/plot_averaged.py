import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from collections import defaultdict

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot DQN results")
    
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory containing logs (with seed folders)")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name")
    parser.add_argument("--output_dir", type=str, default="plots", 
                        help="Directory for saving plots")
    parser.add_argument("--window_size", type=int, default=10, 
                        help="Window size for smoothing")
    parser.add_argument("--alg", type=str, default='dqn', 
                        help="name of algo (dqn, ppo)")
    
    return parser.parse_args()

def smooth(data: List[float], window_size: int = 10) -> np.ndarray:
    """
    Smooth data using a moving average.
    
    Args:
        data: Data to smooth
        window_size: Size of the moving average window
        
    Returns:
        Smoothed data
    """
    if window_size <= 1:
        return np.array(data)
    
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def load_metrics_from_seeds(log_dir: str, env: str, alg: str) -> List[Dict[str, Any]]:
    """
    Load metrics from all seed folders.
    
    Args:
        log_dir: Directory containing seed folders
        env: Environment name
        
    Returns:
        List of metric dictionaries for each seed
    """
    all_metrics = []
    
    # Get all subdirectories (seed folders)
    seed_folders = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    
    if not seed_folders:
        raise FileNotFoundError(f"No seed folders found in {log_dir}")
    
    for seed_folder in seed_folders:
        metrics_path = os.path.join(log_dir, seed_folder, f"{alg}_{env}_metrics.json")
        
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found: {metrics_path}")
            continue
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    if not all_metrics:
        raise FileNotFoundError(f"No metrics found for environment {env}")
    
    return all_metrics

def compute_statistics(all_metrics: List[Dict[str, Any]], window_size: int = 10):
    """
    Compute statistics (mean and standard deviation) across seeds.
    
    Args:
        all_metrics: List of metric dictionaries for each seed
        window_size: Size of the moving average window
        
    Returns:
        Dictionary with statistics
    """
    # Prepare data structures
    max_train_len = max(len(metrics["rewards"]) for metrics in all_metrics)
    max_eval_len = max(len(metrics["eval_rewards"]) for metrics in all_metrics)
    
    # Initialize arrays for train and eval rewards
    all_train_rewards = []
    all_eval_rewards = []
    
    # Process each seed's metrics
    for metrics in all_metrics:
        # Smooth training rewards
        if len(metrics["rewards"]) > window_size:
            smoothed_rewards = smooth(metrics["rewards"], window_size)
            # Pad with NaN if shorter than max length
            padded_rewards = np.full(max_train_len - window_size + 1, np.nan)
            padded_rewards[:len(smoothed_rewards)] = smoothed_rewards
            all_train_rewards.append(padded_rewards)
        else:
            # Handle case where data is shorter than window size
            padded_rewards = np.full(max_train_len, np.nan)
            padded_rewards[:len(metrics["rewards"])] = metrics["rewards"]
            all_train_rewards.append(padded_rewards)
        
        # Process evaluation rewards (no smoothing)
        padded_eval = np.full(max_eval_len, np.nan)
        padded_eval[:len(metrics["eval_rewards"])] = metrics["eval_rewards"]
        all_eval_rewards.append(padded_eval)
    
    # Convert to numpy arrays
    all_train_rewards = np.array(all_train_rewards)
    all_eval_rewards = np.array(all_eval_rewards)
    
    # Compute statistics (ignoring NaN values)
    train_mean = np.nanmean(all_train_rewards, axis=0)
    train_std = np.nanstd(all_train_rewards, axis=0)
    
    eval_mean = np.nanmean(all_eval_rewards, axis=0)
    eval_std = np.nanstd(all_eval_rewards, axis=0)
    
    # Create x-axis values
    x_train = np.arange(window_size - 1, max_train_len) if max_train_len > window_size else np.arange(max_train_len)
    x_eval = np.arange(max_eval_len)
    
    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "x_train": x_train,
        "x_eval": x_eval
    }

def plot_averaged_metrics(stats: Dict[str, np.ndarray], env: str, alg: str, output_dir: str = "plots"):
    """
    Plot averaged metrics with standard deviation regions.
    
    Args:
        stats: Dictionary with statistics
        env: Environment name
        output_dir: Directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    x_eval = np.linspace(0, stats["x_train"][-1], len(stats["x_eval"]))
    
    plt.plot(stats["x_train"], stats["train_mean"], label="Train rewards", color="blue")
    plt.fill_between(
        stats["x_train"],
        stats["train_mean"] - stats["train_std"],
        stats["train_mean"] + stats["train_std"],
        alpha=0.3,
        color="blue",
    )
    plt.title("Train",fontsize=20)
    plt.xlabel("Episode",fontsize=20)
    plt.ylabel("Reward",fontsize=20)
    
    plt.plot(x_eval, stats["eval_mean"], label="Evaluation rewards", color="green")
    plt.fill_between(
        x_eval,
        stats["eval_mean"] - stats["eval_std"],
        stats["eval_mean"] + stats["eval_std"],
        alpha=0.3,
        color="green",
    )
    plt.title("Eval",fontsize=20)
    plt.xlabel("Evaluation",fontsize=20)
    plt.ylabel("Reward",fontsize=20)
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17)
    plt.legend(fontsize=17)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{alg}_{env}_averaged_results.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    
    # Load metrics from all seed folders
    all_metrics = load_metrics_from_seeds(args.log_dir, args.env, args.alg)
    
    # Compute statistics
    stats = compute_statistics(all_metrics, args.window_size)
    
    # Plot averaged metrics
    plot_averaged_metrics(stats, args.env, args.alg, args.output_dir)
    
    print(f"Averaged plots saved to {args.output_dir}/{args.alg}_{args.env}_averaged_results.png")
    print(f"Number of seeds processed: {len(all_metrics)}")