import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot TD3 training results")

    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory containing log files")
    parser.add_argument("--env", type=str, default=None, 
                        help="Environment name (if None, plot all environments)")
    parser.add_argument("--output_dir", type=str, default="plots", 
                        help="Directory to save plots")
    parser.add_argument("--show", action="store_true", 
                        help="Show plots instead of saving them")

    return parser.parse_args()

def load_metrics(log_dir: str, env_name: str) -> Dict[str, List[float]]:
    """
    Load metrics from a JSON file.

    Args:
        log_dir: Directory containing log files
        env_name: Name of the environment

    Returns:
        Dictionary containing metrics
    """
    metrics_path = os.path.join(log_dir, f"td3_{env_name}_metrics.json")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_metrics(metrics: Dict[str, List[float]], env_name: str, output_dir: str, show: bool = False):
    """
    Plot metrics from training.

    Args:
        metrics: Dictionary containing metrics
        env_name: Name of the environment
        output_dir: Directory to save plots
        show: Whether to show plots instead of saving them
    """
    plt.figure(figsize=(12, 8))

    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title(f"{env_name} - Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Plot evaluation rewards
    plt.subplot(2, 2, 2)
    plt.plot(metrics["eval_rewards"])
    plt.title(f"{env_name} - Evaluation Rewards")
    plt.xlabel("Evaluation")
    plt.ylabel("Reward")

    # Plot actor losses
    plt.subplot(2, 2, 3)
    plt.plot(metrics["actor_losses"])
    plt.title(f"{env_name} - Actor Losses")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    # Plot critic losses
    plt.subplot(2, 2, 4)
    plt.plot(metrics["critic_losses"])
    plt.title(f"{env_name} - Critic Losses")
    plt.xlabel("Update")
    plt.ylabel("Loss")

    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"td3_{env_name}_results.png"))
        plt.close()

def plot_all_environments(log_dir: str, output_dir: str, show: bool = False):
    """
    Plot metrics for all environments in the log directory.

    Args:
        log_dir: Directory containing log files
        output_dir: Directory to save plots
        show: Whether to show plots instead of saving them
    """
    # Get all metrics files
    metrics_files = [f for f in os.listdir(log_dir) if f.startswith("td3_") and f.endswith("_metrics.json")]
    
    if not metrics_files:
        print(f"No TD3 metrics files found in {log_dir}")
        return
    
    # Extract environment names
    env_names = [f.split("_")[1] for f in metrics_files]
    
    # Plot metrics for each environment
    for env_name in env_names:
        try:
            metrics = load_metrics(log_dir, env_name)
            plot_metrics(metrics, env_name, output_dir, show)
            print(f"Plotted metrics for {env_name}")
        except Exception as e:
            print(f"Error plotting metrics for {env_name}: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    if args.env is None:
        plot_all_environments(args.log_dir, args.output_dir, args.show)
    else:
        try:
            metrics = load_metrics(args.log_dir, args.env)
            plot_metrics(metrics, args.env, args.output_dir, args.show)
            print(f"Plotted metrics for {args.env}")
        except Exception as e:
            print(f"Error plotting metrics for {args.env}: {e}")

if __name__ == "__main__":
    main()
