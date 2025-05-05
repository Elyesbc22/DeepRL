import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot DQN results")
    
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory containing logs")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name")
    parser.add_argument("--output_dir", type=str, default="plots", 
                        help="Directory for saving plots")
    parser.add_argument("--window_size", type=int, default=10, 
                        help="Window size for smoothing")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare multiple runs")
    parser.add_argument("--runs", type=str, nargs="+", default=None, 
                        help="Runs to compare")
    
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

def load_metrics(log_dir: str, env: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        log_dir: Directory containing logs
        env: Environment name
        
    Returns:
        Dictionary of metrics
    """
    metrics_path = os.path.join(log_dir, f"dqn_{env}_metrics.json")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    return metrics

def plot_metrics(metrics: Dict[str, Any], env: str, window_size: int = 10, output_dir: str = "plots"):
    """
    Plot metrics.
    
    Args:
        metrics: Dictionary of metrics
        env: Environment name
        window_size: Size of the moving average window
        output_dir: Directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Smooth rewards
    if len(metrics["rewards"]) > window_size:
        smoothed_rewards = smooth(metrics["rewards"], window_size)
        x_rewards = np.arange(window_size - 1, len(metrics["rewards"]))
    else:
        smoothed_rewards = metrics["rewards"]
        x_rewards = np.arange(len(metrics["rewards"]))
    
    # Plot rewards
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"], alpha=0.3, label="Raw")
    plt.plot(x_rewards, smoothed_rewards, label=f"Smoothed (window={window_size})")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
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
    plt.savefig(os.path.join(output_dir, f"dqn_{env}_results.png"))
    plt.close()

def compare_runs(log_dir: str, env: str, runs: List[str], window_size: int = 10, output_dir: str = "plots"):
    """
    Compare multiple runs.
    
    Args:
        log_dir: Directory containing logs
        env: Environment name
        runs: List of run names
        window_size: Size of the moving average window
        output_dir: Directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for run in runs:
        metrics_path = os.path.join(log_dir, run, f"dqn_{env}_metrics.json")
        
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found: {metrics_path}")
            continue
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Smooth rewards
        if len(metrics["rewards"]) > window_size:
            smoothed_rewards = smooth(metrics["rewards"], window_size)
            x_rewards = np.arange(window_size - 1, len(metrics["rewards"]))
        else:
            smoothed_rewards = metrics["rewards"]
            x_rewards = np.arange(len(metrics["rewards"]))
        
        plt.plot(x_rewards, smoothed_rewards, label=run)
    
    plt.title(f"Episode Rewards ({env})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dqn_{env}_comparison.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    
    if args.compare:
        if args.runs is None:
            # Find all subdirectories in log_dir
            runs = [d for d in os.listdir(args.log_dir) if os.path.isdir(os.path.join(args.log_dir, d))]
        else:
            runs = args.runs
        
        compare_runs(args.log_dir, args.env, runs, args.window_size, args.output_dir)
    else:
        metrics = load_metrics(args.log_dir, args.env)
        plot_metrics(metrics, args.env, args.window_size, args.output_dir)
