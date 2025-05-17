import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Union, Optional
from collections import deque
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.environment import Environment

class Actor(nn.Module):
    """
    Actor network for TD3 algorithm that maps states to deterministic actions.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        """
        Initialize the Actor network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            max_action: Maximum action value
        """
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Actor network.

        Args:
            state: State tensor

        Returns:
            Action tensor
        """
        # Ensure state has right shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Return scaled action
        return self.max_action * self.net(state)

class Critic(nn.Module):
    """
    Critic network for TD3 algorithm that estimates Q-values.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256
    ):
        """
        Initialize the Critic network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Critic networks.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q1 and Q2 value estimates
        """
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)
        
        # Return Q-values from both critics
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through only the first Critic network.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q1 value estimate
        """
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    def __init__(self, capacity: int = 1_000_000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, s_, d = self.buffer[i]
            states.append(np.array(s, dtype=np.float32))
            actions.append(np.array(a, dtype=np.float32))
            rewards.append(np.array(r, dtype=np.float32))
            next_states.append(np.array(s_, dtype=np.float32))
            dones.append(np.array(d, dtype=np.float32))
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1)
        )
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dim: int = 256,
        buffer_size: int = 1_000_000,
        batch_size: int = 100,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        expl_noise: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the TD3 agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
            hidden_dim: Dimension of hidden layers in networks
            buffer_size: Size of replay buffer
            batch_size: Batch size for updates
            gamma: Discount factor for rewards
            tau: Target network update rate
            policy_noise: Noise added to target policy
            noise_clip: Range to clip noise
            policy_freq: Frequency of delayed policy updates
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic networks
            expl_noise: Exploration noise
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise
        self.device = device
        
        # Initialize actor network and target
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Initialize critic networks and targets
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training variables
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action based on the current state.

        Args:
            state: Current state
            evaluate: Whether to use exploration noise

        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).to(self.device)
        
        # Get deterministic action from actor
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add noise for exploration during training
        if not evaluate:
            noise = np.random.normal(0, self.expl_noise * self.max_action, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action
    
    def update(self) -> Dict[str, float]:
        """
        Update the networks.

        Returns:
            Dictionary with update statistics
        """
        self.total_it += 1
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Select next actions with noise for smoothing
        noise = (
            torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)
        
        next_actions = (
            self.actor_target(next_states) + noise
        ).clamp(-self.max_action, self.max_action)
        
        # Get target Q values
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize the critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = 0
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0
        }
    
    def train(self, env: Environment, episodes: int, max_steps: int = 1000) -> Dict[str, float]:
        """
        Train the agent for a given number of episodes.

        Args:
            env: Environment to train on
            episodes: Number of episodes to train for
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with training statistics
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            
            # Normalize the state
            if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
                state = np.clip(state, env.observation_space.low, env.observation_space.high)
            
            state = np.array(state, dtype=np.float32).flatten()
            
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done and episode_steps < max_steps:
                action = self.select_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Normalize next state
                if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
                    next_state = np.clip(next_state, env.observation_space.low, env.observation_space.high)
                
                next_state = np.array(next_state, dtype=np.float32).flatten()
                
                # Store in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update agent if enough samples
                update_info = {"critic_loss": 0, "actor_loss": 0}
                if len(self.replay_buffer) > self.batch_size:
                    update_info = self.update()
                
                # Update state and tracking variables
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Avg Reward (10) = {avg_reward:.2f}")
        
        return {
            "episode_rewards": episode_rewards,
            "avg_reward": np.mean(episode_rewards)
        }
    
    def save(self, path: str):
        """
        Save the agent's models.

        Args:
            path: Path to save the models to
        """
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """
        Load the agent's models.

        Args:
            path: Path to load the models from
        """
        checkpoint = torch.load(path)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])