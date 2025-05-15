# Twin Delayed Deep Deterministic Policy Gradient (TD3)

This is an implementation of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm as described in the paper "Addressing Function Approximation Error in Actor-Critic Methods" by Fujimoto et al. (2018).

## Features

- Actor-Critic architecture with separate networks
- Twin critics to reduce overestimation bias
- Delayed policy updates for stability
- Target policy smoothing regularization
- Experience replay for sample efficiency
- Support for continuous action spaces
- Configurable network architecture and hyperparameters
- Visualization of training metrics
- Support for OpenAI Gym environments

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym
- tqdm (for progress bars)

## Usage

### Training

To train a TD3 agent on the Pendulum environment:

```bash
python train.py --env Pendulum-v1 --total_timesteps 100000 --actor_lr 1e-3 --critic_lr 1e-3
```

Common command line arguments:

```bash
--env: Environment name (default: Pendulum-v1)
--seed: Random seed (default: 0)
--total_timesteps: Total number of timesteps to train for (default: 1000000)
--batch_size: Batch size for training (default: 100)
--buffer_size: Size of replay buffer (default: 1000000)
--actor_lr: Actor learning rate (default: 3e-4)
--critic_lr: Critic learning rate (default: 3e-4)
--gamma: Discount factor (default: 0.99)
--tau: Target network update rate (default: 0.005)
--policy_noise: Noise added to target policy during critic update (default: 0.2)
--noise_clip: Range to clip target policy noise (default: 0.5)
--policy_freq: Frequency of delayed policy updates (default: 2)
--exploration_noise: Standard deviation of exploration noise (default: 0.1)
--eval_freq: Frequency of evaluation (default: 5000)
--eval_episodes: Number of episodes for evaluation (default: 10)
--hidden_dim: Hidden layer dimension (default: 256)
--log_dir: Directory for saving logs (default: logs)
--save_dir: Directory for saving models (default: saved_models)
--reward_shaping: Enable reward shaping with velocity component (for MountainCar)
--velocity_coefficient: Coefficient for velocity in reward shaping (default: 2)
```

### Visualization

To visualize training metrics:

```bash
python plot.py --log_dir logs --env Pendulum-v1
```

Command line arguments for visualization:

```bash
--log_dir: Directory containing log files (default: logs)
--env: Environment name (if None, plot all environments)
--output_dir: Directory to save plots (default: plots)
--show: Show plots instead of saving them
```

## Supported Environments

TD3 is designed for environments with continuous action spaces. Here are some recommended environments:

- Pendulum-v1
    - Action Space: Continuous (1 dimension)
    - State Space: Continuous (3 dimensions)
    - Goal: Swing up and maintain an inverted pendulum
    - Recommended Hyperparameters:
        - actor_lr: 3e-4
        - critic_lr: 3e-4
        - hidden_dim: 256
        - exploration_noise: 0.1
        - total_timesteps: 500000

Example:

```bash
python train.py --env Pendulum-v1 --total_timesteps 500000 --actor_lr 3e-4 --critic_lr 3e-4 --hidden_dim 256 --exploration_noise 0.1
```

- MountainCarContinuous-v0
    - Action Space: Continuous (1 dimension)
    - State Space: Continuous (2 dimensions)
    - Goal: Drive a car up a mountain with limited momentum
    - Recommended Hyperparameters:
        - actor_lr: 1e-3
        - critic_lr: 1e-3
        - hidden_dim: 256
        - exploration_noise: 0.2
        - reward_shaping: True (to speed up learning)
        - total_timesteps: 300000

Example:

```bash
python train.py --env MountainCarContinuous-v0 --total_timesteps 300000 --actor_lr 1e-3 --critic_lr 1e-3 --hidden_dim 256 --exploration_noise 0.2 --reward_shaping
```

- LunarLanderContinuous-v2
    - Action Space: Continuous (2 dimensions)
    - State Space: Continuous (8 dimensions)
    - Goal: Land a spacecraft on a landing pad
    - Recommended Hyperparameters:
        - actor_lr: 3e-4
        - critic_lr: 3e-4
        - hidden_dim: 256
        - exploration_noise: 0.1
        - total_timesteps: 1000000

Example:

```bash
python train.py --env LunarLanderContinuous-v2 --total_timesteps 1000000 --actor_lr 3e-4 --critic_lr 3e-4 --hidden_dim 256 --exploration_noise 0.1
```

- HalfCheetah-v4
    - Action Space: Continuous (6 dimensions)
    - State Space: Continuous (17 dimensions)
    - Goal: Make a 2D cheetah robot run forward as fast as possible
    - Recommended Hyperparameters:
        - actor_lr: 3e-4
        - critic_lr: 3e-4
        - hidden_dim: 256
        - exploration_noise: 0.1
        - total_timesteps: 1000000

Example:

```bash
python train.py --env HalfCheetah-v4 --total_timesteps 1000000 --actor_lr 3e-4 --critic_lr 3e-4 --hidden_dim 256 --exploration_noise 0.1
```

### Using TD3 with Discrete Action Spaces

TD3 is designed for continuous action spaces, but you can use it with discrete action spaces by using the `ContinuousActionWrapper`:

- CartPole-v1
    - Action Space: Discrete (2 actions)
    - State Space: Continuous (4 dimensions)
    - Goal: Balance a pole on a cart by moving left or right
    - Recommended Hyperparameters:
        - actor_lr: 3e-4
        - critic_lr: 3e-4
        - hidden_dim: 64
        - exploration_noise: 0.1
        - total_timesteps: 200000

Example:

```bash
# First, import the wrapper in your script
from src.env.wrappers import ContinuousActionWrapper

# Then create the wrapped environment
env = ContinuousActionWrapper('CartPole-v1', seed=42)

# Or use it directly in the command line
python train.py --env CartPole-v1 --total_timesteps 200000 --actor_lr 3e-4 --critic_lr 3e-4 --hidden_dim 64 --exploration_noise 0.1
```

- MountainCar-v0
    - Action Space: Discrete (3 actions)
    - State Space: Continuous (2 dimensions)
    - Goal: Drive a car up a mountain with limited momentum
    - Recommended Hyperparameters:
        - actor_lr: 1e-3
        - critic_lr: 1e-3
        - hidden_dim: 128
        - exploration_noise: 0.2
        - reward_shaping: True (to speed up learning)
        - total_timesteps: 300000

Example:

```bash
python train.py --env MountainCar-v0 --total_timesteps 300000 --actor_lr 1e-3 --critic_lr 1e-3 --hidden_dim 128 --exploration_noise 0.2 --reward_shaping
```

- Acrobot-v1
    - Action Space: Discrete (3 actions)
    - State Space: Continuous (6 dimensions)
    - Goal: Swing a two-link robot above a threshold
    - Recommended Hyperparameters:
        - actor_lr: 3e-4
        - critic_lr: 3e-4
        - hidden_dim: 128
        - exploration_noise: 0.1
        - total_timesteps: 500000

Example:

```bash
python train.py --env Acrobot-v1 --total_timesteps 500000 --actor_lr 3e-4 --critic_lr 3e-4 --hidden_dim 128 --exploration_noise 0.1
```



## Algorithm Overview

TD3 addresses the function approximation errors in actor-critic methods through several key mechanisms:

1. **Twin Critics**: Uses two Q-functions to reduce overestimation bias in the Q-function, taking the minimum of the two Q-values during updates.

2. **Delayed Policy Updates**: Updates the policy less frequently than the Q-function to reduce variance and allow the critic to converge to better estimates before the policy is updated.

3. **Target Policy Smoothing**: Adds noise to the target action to make the policy robust to noise and to smooth the value function.

4. **Experience Replay**: Stores and reuses past experiences to improve sample efficiency and break correlations between consecutive samples.

## References

Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. arXiv preprint arXiv:1802.09477.
