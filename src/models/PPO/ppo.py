import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.environment import Environment
from env.utils import set_global_seeds

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        continuous: bool = False,
        action_std_init: float = 0.6  # Initial standard deviation for continuous actions
    ):
        """
        Initialize the Actor-Critic network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space (or number of discrete actions)
            hidden_dim: Dimension of hidden layers
            continuous: Whether the action space is continuous
            action_std_init: Initial action standard deviation for continuous actions
        """
        super(ActorCritic, self).__init__()
        
        self.continuous = continuous
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # For continuous action spaces
        if continuous:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
    
    def set_action_std(self, new_action_std: float):
        """
        Set the standard deviation of actions for continuous action spaces.

        Args:
            new_action_std: New standard deviation value
        """
        if self.continuous:
            self.action_var = torch.full(
                (self.actor[2].out_features,), 
                new_action_std * new_action_std
            )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Action probabilities/means and state value
        """

        # Check if dimensions match
        expected_dim = self.shared[0].weight.shape[1]
        if state.shape[1] != expected_dim:
            raise ValueError(f"State dimension mismatch. Expected {expected_dim}, got {state.shape[1]}. "
                            f"Make sure agent was initialized with correct state_dim.")
        
        features = self.shared(state)
        
        # Actor (policy)
        action_logits = self.actor(features)
        
        # Critic (value function)
        state_value = self.critic(features)
        
        return action_logits, state_value
    
    def act(self, state: torch.Tensor) -> Tuple[Union[int, np.ndarray], torch.Tensor, torch.Tensor]:
        """
        Select an action based on the current state.

        Args:
            state: Current state tensor

        Returns:
            action: Selected action
            action_log_prob: Log probability of selected action
            state_value: Value of the state
        """
        

        with torch.no_grad():
            # Make sure state has the right shape [batch_size, state_dim]
            if len(state.shape) == 1:
                # If state is 1D, add batch dimension
                state = state.unsqueeze(0)
            elif state.shape[0] == self.shared[0].weight.shape[1]:
                # If state shape matches input dimension but is transposed
                state = state.unsqueeze(0)
            elif len(state.shape) == 2 and state.shape[1] == 1:
                # If state has shape [state_dim, 1]
                state = state.transpose(0, 1)
                
            action_logits, state_value = self(state)
            
            if self.continuous:
                action_mean = action_logits
                action_var = self.action_var.to(state.device)
                action_std = torch.sqrt(action_var)
                distribution = Normal(action_mean, action_std)
                
                action = distribution.sample()
                action_log_prob = distribution.log_prob(action).sum(dim=-1)

                # NEW  ➜  strip the batch dimension so env.step() sees shape (action_dim,)
                action_np = action.cpu().numpy().squeeze(0)
                return action_np, action_log_prob, state_value
            else:
                # Get categorical distribution
                action_probs = torch.softmax(action_logits, dim=-1)
                distribution = Categorical(action_probs)
                
                # Sample action and get log probability
                action = distribution.sample()
                action_log_prob = distribution.log_prob(action)
                
                return action.item(), action_log_prob, state_value
    def evaluate(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.

        Args:
            state: Batch of states
            action: Batch of actions

        Returns:
            action_log_probs: Log probabilities of actions
            state_values: Values of the states
            entropy: Entropy of the policy distribution
        """
        action_logits, state_values = self(state)
        
        if self.continuous:
            action_mean = action_logits
            action_var = self.action_var.expand_as(action_mean).to(state.device)
            action_std = torch.sqrt(action_var)
            distribution = Normal(action_mean, action_std)
            
            action_log_probs = distribution.log_prob(action).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
        else:
            action_probs = torch.softmax(action_logits, dim=-1)
            distribution = Categorical(action_probs)
            
            action_log_probs = distribution.log_prob(action)
            entropy = distribution.entropy().mean()
        
        return action_log_probs, state_values, entropy

class PPOBuffer:
    """
    Buffer for storing trajectories collected during training.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        value: torch.Tensor, 
        log_prob: torch.Tensor, 
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Estimated state value
            log_prob: Log probability of the action
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        self.dones.append(done)
    
    def compute_advantages_and_returns(
        self, 
        last_value: float, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).

        Args:
            last_value: Last state value
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: Advantage estimates
            returns: Return estimates
        """
        values = self.values + [last_value]
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        
        last_gae = 0
        for t in reversed(range(len(self.rewards))):
            # If episode terminated, we need to use a value of 0 for the next state
            next_value = values[t + 1] * (1.0 - self.dones[t])
            
            # Compute TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + gamma * next_value - values[t]
            
            # Compute GAE: sum of discounted TD errors
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae
            
            # Compute returns
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data from the buffer.

        Returns:
            states: Batch of states
            actions: Batch of actions
            log_probs: Batch of log probabilities
            advantages: Batch of advantages
            returns: Batch of returns
        """
        # Handle potentially inconsistent state shapes
        try:
            states_array = np.array(self.states)
        except ValueError:
            # If shapes are inconsistent, convert and stack states individually
            states_array = np.vstack([np.array(s, dtype=np.float32).flatten() for s in self.states])
        
        # Handle actions (may be discrete integers or continuous arrays)
        if self.actions and isinstance(self.actions[0], np.ndarray):
            try:
                actions_array = np.array(self.actions)
            except ValueError:
                actions_array = np.vstack([np.array(a, dtype=np.float32).flatten() for a in self.actions])
        else:
            actions_array = np.array(self.actions)
        
        return (
            states_array,
            actions_array,
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.advantages, dtype=np.float32),
            np.array(self.returns, dtype=np.float32)
        )    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

class PPOAgent:
    """
    Proximal Policy Optimization agent.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
        action_std_init: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the PPO agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers in the networks
            lr: Learning rate
            gamma: Discount factor for rewards
            gae_lambda: GAE lambda parameter for advantage estimation
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum norm of gradients for clipping
            continuous: Whether the action space is continuous
            action_std_init: Initial action standard deviation (for continuous actions)
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        self.device = device
        # Initialize actor-critic network
        self.policy = ActorCritic(
            state_dim, 
            action_dim, 
            hidden_dim, 
            continuous,
            action_std_init
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize buffer
        self.buffer = PPOBuffer()
    
    def set_action_std(self, new_action_std: float):
        """
        Set action standard deviation for continuous action spaces.

        Args:
            new_action_std: New standard deviation value
        """
        if self.continuous:
            self.policy.set_action_std(new_action_std)
    
    def select_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Select an action based on the current state.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.policy.act(state)
        return action
    
    def collect_trajectory(self, env: Environment, steps: int = 2048) -> Dict[str, float]:
        """
        Collect a trajectory by running the environment.

        Args:
            env: Environment to collect trajectory from
            steps: Number of steps to collect

        Returns:
            Dictionary with episode statistics
        """
        state, _ = env.reset()

        # Normalize the state if needed
        if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
            state = np.clip(state, env.observation_space.low, env.observation_space.high)
            
        state = np.array(state, dtype=np.float32).flatten()

        done = False
        episode_reward = 0
        episode_length = 0
        episodes = 0
        total_reward = 0
        
        for _ in range(steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Select action
            action, log_prob, value = self.policy.act(state_tensor)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
                next_state = np.clip(next_state, env.observation_space.low, env.observation_space.high)
            
            next_state = np.array(next_state, dtype=np.float32).flatten()          
              
            done = terminated or truncated
            
            # Store experience in buffer
            self.buffer.add(state, action, reward, value, log_prob, done)
            
            # Update tracking variables
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # If episode is done, reset environment
            if done:
                episodes += 1
                total_reward += episode_reward
                
                # Reset environment
                state, _ = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
        
        # Get final value (for bootstrapping)
        if not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, last_value = self.policy(state_tensor)
            last_value = last_value.cpu().item()
        else:
            last_value = 0
        
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_advantages_and_returns(
            last_value, self.gamma, self.gae_lambda
        )
        self.buffer.advantages = advantages
        self.buffer.returns = returns
        
        # Calculate statistics
        avg_reward = total_reward / max(1, episodes)
        
        return {
            "episodes": episodes,
            "total_reward": total_reward,
            "avg_reward": avg_reward
        }
    
    def update(self, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Update the policy using collected trajectories.

        Args:
            epochs: Number of optimization epochs per update
            batch_size: Minibatch size for updates

        Returns:
            Dictionary with update statistics
        """
        # Get data from buffer
        states, actions, old_log_probs, advantages, returns = self.buffer.get()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(actions).to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track losses
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Perform updates for multiple epochs
        for _ in range(epochs):
            # Create dataloader for minibatch updates
            dataset_size = len(states)
            indices = np.random.permutation(dataset_size)
            
            # Minibatch updates
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions under current policy
                new_log_probs, state_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                # Compute ratio between new and old policy
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Policy loss (negative because we're minimizing)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (state_values.squeeze() - batch_returns).pow(2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear buffer
        self.buffer.clear()
        
        # Number of updates (epochs * num_minibatches)
        num_updates = epochs * ((len(states) + batch_size - 1) // batch_size)
        
        # Return average losses
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }
    
    def save(self, path: str):
        """
        Save the agent's model.

        Args:
            path: Path to save the model to
        """
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """
        Load the agent's model.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])