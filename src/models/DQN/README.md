# Deep Q-Network (DQN)

This is an implementation of the Deep Q-Network (DQN) algorithm as described in the paper "Human-level control through deep reinforcement learning" by Mnih et al. (2015).

## Features

- Deep Q-Network with experience replay and target network
- Support for discrete action spaces
- Configurable network architecture and hyperparameters
- Training and evaluation on OpenAI Gym environments
- Plotting and visualization of training results

## Usage

### Training

To train a DQN agent on the CartPole environment:

```bash
python train.py --env CartPole-v1 --total_timesteps 100000
```

You can customize the training process with various command-line arguments:

```bash
python train.py --env CartPole-v1 --total_timesteps 100000 --batch_size 64 --buffer_size 10000 --learning_rate 1e-3 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995 --target_update_freq 10 --eval_freq 1000 --eval_episodes 10 --hidden_dim 128 --log_dir logs --save_dir saved_models
```

Alternatively, you can specify a configuration file:

```bash
python train.py --config config.yaml
```

### Plotting

To plot the results of a training run:

```bash
python plot.py --log_dir logs --env CartPole-v1 --output_dir plots --window_size 10
```

To compare multiple runs:

```bash
python plot.py --log_dir logs --env CartPole-v1 --output_dir plots --window_size 10 --compare --runs run1 run2 run3
```

## Implementation Details

### DQN Algorithm

The DQN algorithm combines Q-learning with deep neural networks to learn policies from high-dimensional inputs. Key components include:

1. **Experience Replay**: Stores transitions in a replay buffer and samples random batches for training, breaking correlations in the observation sequence.
2. **Target Network**: Uses a separate target network for generating TD targets, updated periodically to stabilize training.
3. **Epsilon-Greedy Exploration**: Balances exploration and exploitation by selecting random actions with probability epsilon.

### Network Architecture

The Q-network consists of:
- Input layer with size equal to the state dimension
- Two hidden layers with ReLU activations
- Output layer with size equal to the action dimension

### Training Process

1. Initialize the Q-network and target network
2. For each step:
   - Select an action using epsilon-greedy policy
   - Execute the action and observe the next state and reward
   - Store the transition in the replay buffer
   - Sample a random batch from the replay buffer
   - Compute the target Q-values using the target network
   - Update the Q-network to minimize the loss between current and target Q-values
   - Periodically update the target network
   - Decay epsilon

## References

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
