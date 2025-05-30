# Deep Q-Network (DQN)

This is an implementation of the Deep Q-Network (DQN) algorithm as described in the paper "Human-level control through deep reinforcement learning" by Mnih et al. (2015).

## Features

- Deep Q-Network with experience replay and target network
- Support for discrete action spaces
- Support for continuous action spaces through discretization
- Configurable network architecture and hyperparameters
- Hyperparameter search (grid search, random search, Bayesian optimization, and evolutionary search)
- Training and evaluation on multiple OpenAI Gym environments (CartPole, MountainCar, MountainCarContinuous, Acrobot, and Pendulum)
- Plotting and visualization of training results
- Comparative analysis across different environments
- GIF visualizations of trained agents' behavior in all environments

## Usage

### Training

To train a DQN agent on the CartPole environment:

```bash
python train.py --env CartPole-v1 --total_timesteps 100000
```

You can customize the training process with various command-line arguments:

```bash
python train.py --env CartPole-v1 --total_timesteps 100_000 --batch_size 64 --buffer_size 100_000 --learning_rate 2.3e-3 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.04 --epsilon_decay 0.99 --epsilon_decay_type exponential --target_update_freq 256 --eval_freq 1000 --eval_episodes 10 --hidden_dim 128 --log_dir logs --save_dir saved_models
python train.py --env MountainCarContinuous-v0 --reward-shaping --total_timesteps 200_000 --epsilon_decay_type linear --batch_size 64 --buffer_size 10_000 --learning_rate 1e-4 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.9999 --target_update_freq 256 --eval_freq 500 --eval_episodes 5 --hidden_dim 256 --log_dir logs --save_dir saved_models
python train.py --env MountainCarContinuous-v0 --total_timesteps 200_000 --epsilon_decay_type linear --batch_size 64 --buffer_size 10_000 --learning_rate 1e-4 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.9999 --target_update_freq 256 --eval_freq 500 --eval_episodes 5 --hidden_dim 256 --log_dir logs --save_dir saved_models
python train.py --env MountainCarContinuous-v0 --total_timesteps 200_000 --epsilon_decay_type exponential --batch_size 128 --buffer_size 10_000 --learning_rate 4e-3 --gamma 0.98 --epsilon_start 1.0 --epsilon_end 0.07 --epsilon_decay 0.9999 --target_update_freq 600 --eval_freq 500 --eval_episodes 5 --hidden_dim 256 --log_dir logs --save_dir saved_models
```

Alternatively, you can specify a configuration file:

```bash
python train.py --config config.yaml
```

### Multi-Environment Training

To train a DQN agent on all supported environments (CartPole, MountainCar, MountainCarContinuous, Acrobot, and Pendulum):

```bash
python run_all_envs.py --total_timesteps 100000
```

This script will:
1. Run DQN on each environment sequentially
2. Use discretization for continuous action spaces (MountainCarContinuous and Pendulum)
3. Save metrics and models for each environment
4. Create visualizations comparing performance across environments

You can use the same command-line arguments as with the `train.py` script to customize the training process:

```bash
python run_all_envs.py --total_timesteps 100000 --batch_size 64 --buffer_size 10000 --learning_rate 1e-3 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995 --epsilon_decay_type exponential --target_update_freq 10 --eval_freq 1000 --eval_episodes 10 --hidden_dim 128 --log_dir logs --save_dir saved_models
```

You can also use linear epsilon decay instead of exponential:

```bash
python run_all_envs.py --total_timesteps 100000 --batch_size 64 --buffer_size 10000 --learning_rate 1e-3 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay_type linear --target_update_freq 10 --eval_freq 1000 --eval_episodes 10 --hidden_dim 128 --log_dir logs --save_dir saved_models
```

The script will generate two additional visualizations:
1. `dqn_all_environments_comparison.png`: A comparison of evaluation rewards across all environments
2. `dqn_all_environments_training.png`: Training rewards for each environment in separate subplots

### Visualization

To visualize the behavior of trained DQN agents in all supported environments:

```bash
python visualize.py --model_dir saved_models --vis_dir visualizations
```

This script will:
1. Load the latest trained model for each environment
2. Run the agent in the environment with rendering enabled
3. Save the rendered frames as a GIF
4. Store all GIFs in the specified visualization directory

You can customize the visualization process with various command-line arguments:

```bash
python visualize.py --model_dir saved_models --vis_dir visualizations --num_episodes 3 --fps 60 --seed 42
```

Available options:
- `--model_dir`: Directory containing trained models (default: ./models)
- `--vis_dir`: Directory to save visualizations (default: ./visualizations)
- `--num_episodes`: Number of episodes to visualize (default: 1)
- `--fps`: Frames per second in the output GIF (default: 30)
- `--seed`: Random seed for reproducibility (default: 42)

The script will generate a GIF for each environment, showing the trained agent's behavior. These visualizations provide an intuitive understanding of what the agent has learned and how it interacts with the environment.

### Hyperparameter Search

To perform a hyperparameter search for the DQN agent:

```bash
python hyperparam_search.py --env CartPole-v1 --search_type grid
python hyperparam_search.py --env MountainCarContinuous-v0 --search_type evolutionary
```

You can customize the hyperparameter search with various command-line arguments:

```bash
python hyperparam_search.py --env CartPole-v1 --search_type random --num_trials 20 --total_timesteps 50000 --seed 42 --log_dir logs/my_search --save_dir saved_models/my_search --output_dir plots/my_search
```

Available options:
- `--search_type`: Type of hyperparameter search (`grid`, `random`, `bayesian`, or `evolutionary`)
- `--num_trials`: Number of trials (for random, bayesian, and evolutionary search)
- `--population_size`: Population size for evolutionary search (default: 10)
- `--num_generations`: Number of generations for evolutionary search (default: 5)
- `--mutation_rate`: Mutation rate for evolutionary search (default: 0.1)
- `--total_timesteps`: Total number of timesteps to train each configuration for
- `--log_dir`: Directory for saving logs
- `--save_dir`: Directory for saving models
- `--output_dir`: Directory for saving plots and results

#### Search Methods

**Grid Search**
```bash
python hyperparam_search.py --env CartPole-v1 --search_type grid
```
Grid search exhaustively evaluates all possible combinations of hyperparameters. This is thorough but becomes computationally expensive as the number of hyperparameters and their possible values increase.

**Random Search**
```bash
python hyperparam_search.py --env CartPole-v1 --search_type random --num_trials 20
```
Random search samples random configurations from the hyperparameter space. This can be more efficient than grid search when some hyperparameters have little impact on performance.

**Bayesian Optimization**
```bash
python hyperparam_search.py --env CartPole-v1 --search_type bayesian --num_trials 20
```
Bayesian optimization builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters to evaluate next. This approach balances exploration (trying configurations with high uncertainty) and exploitation (trying configurations with high predicted performance).

**Evolutionary Search**
```bash
python hyperparam_search.py --env CartPole-v1 --search_type evolutionary --population_size 10 --num_generations 5 --mutation_rate 0.1
```
Evolutionary search uses principles from genetic algorithms to evolve a population of hyperparameter configurations. It starts with a random population and evolves it over multiple generations using selection, crossover, and mutation operations.

The hyperparameter search will try different combinations of:
- Learning rate
- Discount factor (gamma)
- Exploration rate decay
- Exploration rate decay type (exponential or linear)
- Hidden layer dimensions
- Batch size
- Replay buffer size
- Target network update frequency

After the search is complete, the script will print the results, save them to a JSON file, and create a plot comparing the performance of different configurations.

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
3. **Epsilon-Greedy Exploration**: Balances exploration and exploitation by selecting random actions with probability epsilon. Two types of epsilon decay are supported:
   - **Exponential Decay**: Multiplies epsilon by a decay factor at each step (epsilon = epsilon * decay_factor)
   - **Linear Decay**: Linearly decreases epsilon from start to end value over a large portion (90%) of the total training steps

### Handling Continuous Action Spaces

DQN is designed for discrete action spaces, but this implementation includes support for continuous action spaces through discretization:

1. **Discretized Action Wrapper**: A wrapper class that converts continuous action spaces into discrete ones by dividing each action dimension into a fixed number of bins.
2. **Action Mapping**: When a discrete action is selected, it is mapped back to a continuous value within the original action space.
3. **Multi-dimensional Actions**: For environments with multi-dimensional action spaces, the discretization is applied to each dimension independently.

This approach allows DQN to be applied to environments with continuous action spaces like MountainCarContinuous and Pendulum, although it may not be as effective as algorithms specifically designed for continuous control (like DDPG or SAC).

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

### Best Hyperparameters Management

This implementation includes a mechanism to store and update the best hyperparameters for each environment:

1. **Storage**: The best hyperparameters for each environment are stored in a JSON file (`best_hyperparams.json`) in the DQN directory.
2. **Automatic Usage**: When training a new agent, the system automatically checks if there are best hyperparameters available for the target environment. If available, these hyperparameters are used instead of the default ones or those provided via command-line arguments.
3. **Continuous Improvement**: During training, whenever the agent is evaluated (both during periodic evaluations and at the end of training), the current hyperparameters are compared with the stored best hyperparameters. If the current evaluation results are better, the stored hyperparameters are updated.

This approach ensures that:
- Each environment uses the most effective hyperparameters discovered so far
- The hyperparameters continuously improve as better configurations are found
- New training runs benefit from previous discoveries without manual intervention

To see the current best hyperparameters for an environment, you can examine the `best_hyperparams.json` file. The file structure is:

```json
{
  "CartPole-v1": {
    "hidden_dim": 128,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "epsilon_decay_type": "exponential",
    "buffer_size": 10000,
    "batch_size": 64,
    "target_update_freq": 10,
    "eval_reward": 500.0
  },
  "MountainCar-v0": {
    ...
  }
}
```

Each environment entry includes all the hyperparameters and the evaluation reward achieved with those hyperparameters.

## References

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
