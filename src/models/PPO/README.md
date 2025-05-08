# Proximal Policy Optimization (PPO)

This is an implementation of the Proximal Policy Optimization (PPO) algorithm as described in the paper "Proximal Policy Optimization Algorithms" by Schulman et al. (2017).

## Features

- Actor-Critic architecture with shared feature extraction
- Support for both discrete and continuous action spaces
- Policy gradient learning with clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Configurable network architecture and hyperparameters
- Hyperparameter search (grid, random, Bayesian, and evolutionary)
- Visualization of trained agents
- Support for OpenAI Gym environments

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym
- imageio (for GIF creation)

## Usage

### Training

To train a PPO agent on the CartPole environment:

```bash
python train.py --env CartPole-v1 --total_timesteps 500000
```

Common command line arguments:

```bash
--env: Environment name (default: CartPole-v1)
--seed: Random seed (default: 0)
--total_timesteps: Total number of timesteps to train for (default: 1000000)
--learning_rate: Learning rate (default: 3e-4)
--gamma: Discount factor (default: 0.99)
--gae_lambda: GAE lambda parameter (default: 0.95)
--clip_ratio: PPO clipping parameter (default: 0.2)
--value_coef: Value loss coefficient (default: 0.5)
--entropy_coef: Entropy bonus coefficient (default: 0.01)
--steps_per_update: Number of steps to collect before update (default: 2048)
--epochs_per_update: Number of optimization epochs per update (default: 10)
--batch_size: Minibatch size (default: 64)
--hidden_dim: Hidden layer dimension (default: 64)
--max_grad_norm: Maximum gradient norm (default: 0.5)
--log_dir: Directory for saving logs (default: logs)
--save_dir: Directory for saving models (default: saved_models)
```

Hyperparameter Search
To find optimal hyperparameters for a specific environment:

```bash
python hyperparam_search.py --env CartPole-v1 --search_type random --num_trials 20
```

Search types:
```bash
grid: Exhaustive search over all combinations
random: Random sampling of configurations
bayesian: Bayesian optimization for efficient search
evolutionary: Genetic algorithm-based search
```

### Visualization

To visualize trained agents:

```bash
python visualize.py --model_dir saved_models --vis_dir visualizations
```

This will generate GIF animations of the trained agents acting in their environments.

Supported Environments
- CartPole-v1
    - Action Space: Discrete (2 actions)
    - State Space: Continuous (4 dimensions)
    - Goal: Balance a pole on a cart by moving left or right

- Recommended Hyperparameters:
    - learning_rate: 3e-4
    - hidden_dim: 64
    - steps_per_update: 2048

Example:

```bash
python train.py --env CartPole-v1 --total_timesteps 200000 --learning_rate 3e-4 --hidden_dim 64
```

- MountainCar-v0
    - Action Space: Discrete (3 actions)
    - State Space: Continuous (2 dimensions)
    - Goal: Drive a car up a mountain with limited momentum
    - Challenge: Sparse rewards make this environment difficult
- Recommended Hyperparameters:
    - learning_rate: 1e-3
    - gamma: 0.99
    - entropy_coef: 0.05
    - steps_per_update: 4096

Example:

```bash
python train.py --env MountainCar-v0 --total_timesteps 500000 --learning_rate 1e-3 --entropy_coef 0.05
```

- MountainCarContinuous-v0
    - Action Space: Continuous (1 dimension)
    - State Space: Continuous (2 dimensions)
    - Goal: Same as MountainCar but with continuous force control
- Recommended Hyperparameters:
    - learning_rate: 3e-4
    - steps_per_update: 2048
    - epochs_per_update: 10

Example:

```bash
python train.py --env MountainCar-v0 --total_timesteps 500000 --learning_rate 1e-3 --entropy_coef 0.05
```

- Acrobot-v1
    - Action Space: Discrete (3 actions)
    - State Space: Continuous (6 dimensions)
    - Goal: Swing a two-link robot above a threshold
- Recommended Hyperparameters:
    - learning_rate: 3e-4
    - hidden_dim: 128
    - value_coef: 0.5

Example:

```bash
python train.py --env Acrobot-v1 --total_timesteps 500000 --hidden_dim 128
```

- Action Space: Continuous (1 dimension)
    - State Space: Continuous (3 dimensions)
    - Goal: Swing up and maintain an inverted pendulum
- Recommended Hyperparameters:
    - learning_rate: 3e-4
    - steps_per_update: 1024
    - clip_ratio: 0.2

Example:

```bash
python train.py --env Acrobot-v1 --total_timesteps 500000 --hidden_dim 128
```

- LunarLander-v2
    - Action Space: Discrete (4 actions)
    - State Space: Continuous (8 dimensions)
    - Goal: Land a spacecraft on a landing pad
- Recommended Hyperparameters:
    - learning_rate: 1e-4
    - hidden_dim: 256
    - steps_per_update: 4096

Example:

```bash
python train.py --env LunarLander-v2 --total_timesteps 1000000 --hidden_dim 256 --steps_per_update 4096
```

Algorithm Overview
PPO combines the benefits of on-policy and off-policy methods by using a clipped surrogate objective:

1. Actor-Critic Architecture: Simultaneously learns a policy (actor) and a value function (critic)
2. Clipped Surrogate Objective: Prevents too large policy updates that may destabilize training
3. Generalized Advantage Estimation: Reduces variance in policy gradient estimation

References
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.
