import os
import argparse
import itertools
import random
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
import copy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for DQN")

    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=100000, 
                        help="Total number of timesteps to train for")
    parser.add_argument("--search_type", type=str, 
                        choices=["grid", "random", "bayesian", "evolutionary"], 
                        default="grid",
                        help="Type of hyperparameter search (grid, random, bayesian, or evolutionary)")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials (for random, bayesian, and evolutionary search)")
    parser.add_argument("--population_size", type=int, default=10,
                        help="Population size for evolutionary search")
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Number of generations for evolutionary search")
    parser.add_argument("--mutation_rate", type=float, default=0.1,
                        help="Mutation rate for evolutionary search")
    parser.add_argument("--log_dir", type=str, default="logs/hyperparam_search",
                        help="Directory for saving logs")
    parser.add_argument("--save_dir", type=str, default="saved_models/hyperparam_search",
                        help="Directory for saving models")
    parser.add_argument("--output_dir", type=str, default="plots/hyperparam_search",
                        help="Directory for saving plots")

    return parser.parse_args()

def define_search_space() -> Dict[str, List[Any]]:
    """
    Define the hyperparameter search space.

    Returns:
        Dictionary mapping hyperparameter names to lists of possible values
    """
    search_space = {
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
        "gamma": [0.9, 0.95, 0.99],
        "epsilon_decay": [0.99, 0.995, 0.999],
        "hidden_dim": [64, 128, 256],
        "batch_size": [32, 64, 128],
        "buffer_size": [5000, 10000, 20000],
        "target_update_freq": [5, 10, 20]
    }

    return search_space

def grid_search(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters for grid search.

    Args:
        search_space: Dictionary mapping hyperparameter names to lists of possible values

    Returns:
        List of dictionaries, each containing a hyperparameter configuration
    """
    keys = search_space.keys()
    values = search_space.values()

    configurations = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configurations.append(config)

    return configurations

def random_search(search_space: Dict[str, List[Any]], num_trials: int) -> List[Dict[str, Any]]:
    """
    Generate random hyperparameter configurations for random search.

    Args:
        search_space: Dictionary mapping hyperparameter names to lists of possible values
        num_trials: Number of random configurations to generate

    Returns:
        List of dictionaries, each containing a hyperparameter configuration
    """
    configurations = []

    for _ in range(num_trials):
        config = {}
        for key, values in search_space.items():
            config[key] = random.choice(values)
        configurations.append(config)

    return configurations

def run_training(config: Dict[str, Any], args: argparse.Namespace, run_id: str) -> str:
    """
    Run training with a specific hyperparameter configuration.

    Args:
        config: Hyperparameter configuration
        args: Command line arguments
        run_id: Unique identifier for this run

    Returns:
        Path to the metrics file
    """
    # Create run directory
    run_dir = os.path.join(args.log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Create save directory
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # Build command
    cmd = [
        "python", "train.py",
        "--env", args.env,
        "--seed", str(args.seed),
        "--total_timesteps", str(args.total_timesteps),
        "--learning_rate", str(config["learning_rate"]),
        "--gamma", str(config["gamma"]),
        "--epsilon_decay", str(config["epsilon_decay"]),
        "--hidden_dim", str(config["hidden_dim"]),
        "--batch_size", str(config["batch_size"]),
        "--buffer_size", str(config["buffer_size"]),
        "--target_update_freq", str(config["target_update_freq"]),
        "--log_dir", run_dir,
        "--save_dir", save_dir
    ]

    # Run command
    print(f"Running training with configuration: {config}")
    subprocess.run(cmd, check=True)

    # Return path to metrics file
    return os.path.join(run_dir, f"dqn_{args.env}_metrics.json")

def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.

    Args:
        metrics_path: Path to the metrics file

    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics

def evaluate_configuration(metrics: Dict[str, Any]) -> float:
    """
    Evaluate a hyperparameter configuration based on its metrics.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Score for the configuration (higher is better)
    """
    # Use the mean of the last 10 evaluation rewards as the score
    eval_rewards = metrics["eval_rewards"]
    if len(eval_rewards) < 10:
        score = np.mean(eval_rewards)
    else:
        score = np.mean(eval_rewards[-10:])

    return score

def compare_configurations(results: List[Tuple[Dict[str, Any], float]], args: argparse.Namespace):
    """
    Compare different hyperparameter configurations.

    Args:
        results: List of (configuration, score) tuples
        args: Command line arguments
    """
    # Sort configurations by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print results
    print("\nHyperparameter Search Results:")
    print("==============================")

    for i, (config, score) in enumerate(results):
        print(f"Rank {i+1}: Score = {score:.2f}")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

    # Save results to file
    results_path = os.path.join(args.output_dir, f"hyperparam_search_results_{args.env}.json")
    with open(results_path, "w") as f:
        json.dump({
            "results": [{"config": config, "score": score} for config, score in results],
            "best_config": results[0][0],
            "best_score": results[0][1]
        }, f, indent=2)

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot scores
    scores = [score for _, score in results]
    plt.bar(range(len(scores)), scores)
    plt.title(f"Hyperparameter Search Results ({args.env})")
    plt.xlabel("Configuration Rank")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"hyperparam_search_results_{args.env}.png"))
    plt.close()

def bayesian_search(search_space: Dict[str, List[Any]], num_trials: int, 
                     objective_func: Callable[[Dict[str, Any]], float]) -> List[Dict[str, Any]]:
    """
    Perform Bayesian optimization for hyperparameter search.

    Args:
        search_space: Dictionary mapping hyperparameter names to lists of possible values
        num_trials: Number of trials to run
        objective_func: Function that evaluates a configuration and returns a score

    Returns:
        List of dictionaries, each containing a hyperparameter configuration
    """
    # Initialize with random configurations
    configurations = random_search(search_space, 2)
    scores = []

    # Convert hyperparameters to indices for GP modeling
    param_indices = {}
    for key, values in search_space.items():
        param_indices[key] = {value: i for i, value in enumerate(values)}

    for trial in range(num_trials - 2):  # We already have 2 initial configurations
        if trial > 0:  # Skip for the first iteration as we don't have scores yet
            # Evaluate all configurations so far
            X = []
            for config in configurations:
                x = [param_indices[key][config[key]] / (len(values) - 1) for key, values in search_space.items()]
                X.append(x)
            X = np.array(X)
            y = np.array(scores)

            # Normalize y for better GP performance
            y_mean, y_std = np.mean(y), np.std(y) if np.std(y) > 0 else 1.0
            y_norm = (y - y_mean) / y_std

            # Generate candidate points (all possible configurations)
            candidates = grid_search(search_space)
            candidate_X = []
            for config in candidates:
                if config not in configurations:  # Only consider configurations we haven't tried
                    x = [param_indices[key][config[key]] / (len(values) - 1) for key, values in search_space.items()]
                    candidate_X.append((x, config))

            if not candidate_X:  # If we've tried all configurations, break
                break

            # Compute expected improvement for each candidate
            best_ei = -1
            next_config = None

            for x, config in candidate_X:
                x = np.array(x).reshape(1, -1)

                # Simple GP prediction (mean and variance)
                # In a real implementation, you'd use a proper GP library
                if len(X) > 1:
                    dists = np.sum((X[:, np.newaxis, :] - x) ** 2, axis=2)
                    weights = np.exp(-dists)
                    mean = np.sum(weights * y_norm) / np.sum(weights) if np.sum(weights) > 0 else 0
                    var = np.mean((y_norm - mean) ** 2) + 1e-6  # Add small constant for numerical stability
                else:
                    mean, var = 0, 1.0

                # Expected improvement
                z = (mean - np.max(y_norm)) / np.sqrt(var)
                ei = (mean - np.max(y_norm)) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

                if ei > best_ei:
                    best_ei = ei
                    next_config = config

            if next_config is not None:
                configurations.append(next_config)

        # Evaluate the configuration
        score = objective_func(configurations[-1])
        scores.append(score)

        print(f"Bayesian optimization trial {trial+3}/{num_trials} completed. Score: {score:.2f}")

    return configurations

def evolutionary_search(search_space: Dict[str, List[Any]], population_size: int, 
                       num_generations: int, mutation_rate: float,
                       objective_func: Callable[[Dict[str, Any]], float]) -> List[Dict[str, Any]]:
    """
    Perform evolutionary search for hyperparameter optimization.

    Args:
        search_space: Dictionary mapping hyperparameter names to lists of possible values
        population_size: Size of the population
        num_generations: Number of generations
        mutation_rate: Probability of mutation
        objective_func: Function that evaluates a configuration and returns a score

    Returns:
        List of dictionaries, each containing a hyperparameter configuration
    """
    # Initialize population with random configurations
    population = random_search(search_space, population_size)
    all_configurations = []
    all_configurations.extend(population)

    # Evaluate initial population
    fitness_scores = []
    for config in population:
        score = objective_func(config)
        fitness_scores.append(score)
        print(f"Initial population member evaluated. Score: {score:.2f}")

    for generation in range(num_generations):
        print(f"Generation {generation+1}/{num_generations}")

        # Select parents (tournament selection)
        parents = []
        for _ in range(population_size):
            # Select two random individuals
            idx1, idx2 = random.sample(range(population_size), 2)
            # Select the one with higher fitness
            if fitness_scores[idx1] > fitness_scores[idx2]:
                parents.append(copy.deepcopy(population[idx1]))
            else:
                parents.append(copy.deepcopy(population[idx2]))

        # Create next generation through crossover and mutation
        next_generation = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                parent1, parent2 = parents[i], parents[i+1]

                # Crossover
                child1, child2 = {}, {}
                for key in search_space.keys():
                    if random.random() < 0.5:
                        child1[key] = parent1[key]
                        child2[key] = parent2[key]
                    else:
                        child1[key] = parent2[key]
                        child2[key] = parent1[key]

                # Mutation
                for child in [child1, child2]:
                    for key, values in search_space.items():
                        if random.random() < mutation_rate:
                            child[key] = random.choice(values)

                next_generation.extend([child1, child2])
            else:
                # If odd number of parents, just add the last one
                next_generation.append(parents[i])

        # Evaluate new generation
        population = next_generation
        all_configurations.extend(population)
        fitness_scores = []

        for config in population:
            score = objective_func(config)
            fitness_scores.append(score)
            print(f"Generation {generation+1} member evaluated. Score: {score:.2f}")

    return all_configurations

def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Define search space
    search_space = define_search_space()

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Define objective function
    def objective_function(config):
        # Generate a unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{len(results)}_{timestamp}"

        # Run training
        metrics_path = run_training(config, args, run_id)

        # Load metrics
        metrics = load_metrics(metrics_path)

        # Evaluate configuration
        return evaluate_configuration(metrics)

    # Generate configurations and run search
    results = []

    if args.search_type == "grid":
        configurations = grid_search(search_space)
        print(f"Generated {len(configurations)} configurations for grid search.")

        # Run training for each configuration
        for i, config in enumerate(configurations):
            # Run training and get score
            score = objective_function(config)

            # Store results
            results.append((config, score))

            print(f"Configuration {i+1}/{len(configurations)} completed. Score: {score:.2f}")

    elif args.search_type == "random":
        configurations = random_search(search_space, args.num_trials)
        print(f"Generated {len(configurations)} configurations for random search.")

        # Run training for each configuration
        for i, config in enumerate(configurations):
            # Run training and get score
            score = objective_function(config)

            # Store results
            results.append((config, score))

            print(f"Configuration {i+1}/{len(configurations)} completed. Score: {score:.2f}")

    elif args.search_type == "bayesian":
        print(f"Starting Bayesian optimization with {args.num_trials} trials.")

        # Define a wrapper function that stores results
        def objective_wrapper(config):
            score = objective_function(config)
            results.append((config, score))
            return score

        # Run Bayesian optimization
        bayesian_search(search_space, args.num_trials, objective_wrapper)

    elif args.search_type == "evolutionary":
        print(f"Starting evolutionary search with population size {args.population_size} for {args.num_generations} generations.")

        # Define a wrapper function that stores results
        def objective_wrapper(config):
            score = objective_function(config)
            results.append((config, score))
            return score

        # Run evolutionary search
        evolutionary_search(search_space, args.population_size, args.num_generations, 
                           args.mutation_rate, objective_wrapper)

    # Compare configurations
    compare_configurations(results, args)

if __name__ == "__main__":
    main()
