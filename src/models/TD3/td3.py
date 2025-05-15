import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any, Optional

class Actor(nn.Module):
    """
    Actor network for TD3 algorithm.
    Maps states to deterministic actions.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        """
        Initialize the actor network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
            hidden_dim: Dimension of the hidden layers
        """
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
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
        Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Action tensor scaled by max_action
        """
        return self.max_action * self.network(state)

class Critic(nn.Module):
    """
    Critic network for TD3 algorithm.
    Maps state-action pairs to Q-values.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the critic network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
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
        Forward pass through both critic networks.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q-values from both critics
        """
        sa = torch.cat([state, action], 1)
        
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through only the first critic network.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q-value from the first critic
        """
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.

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
        Sample a batch of experiences from the buffer.

        Args:
            batch_size: Size of the batch to sample

        Returns:
            Batch of states, actions, rewards, next_states, dones
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
        
    def __len__(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            Current size of the buffer
        """
        return len(self.buffer)

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        buffer_size: int = 1000000,
        batch_size: int = 100,
        exploration_noise: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the TD3 agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_action: Maximum action value
            hidden_dim: Dimension of the hidden layers
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            gamma: Discount factor
            tau: Target network update rate
            policy_noise: Noise added to target policy during critic update
            noise_clip: Range to clip target policy noise
            policy_freq: Frequency of delayed policy updates
            buffer_size: Size of the replay buffer
            batch_size: Size of the batch for training
            exploration_noise: Standard deviation of exploration noise
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise * max_action
        self.device = device
        
        # Initialize actor networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.total_it = 0
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select an action using the policy with noise for exploration.

        Args:
            state: Current state
            eval_mode: Whether to use evaluation mode (no exploration)

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if not eval_mode:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action
    
    def update(self) -> Dict[str, float]:
        """
        Update the networks using a batch of experiences.

        Returns:
            Dictionary with loss information
        """
        self.total_it += 1
        
        if len(self.replay_buffer) < self.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(actions_tensor) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (
                self.actor_target(next_states_tensor) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_states_tensor, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states_tensor, actions_tensor)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = 0.0
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(states_tensor, self.actor(states_tensor)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {"actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss, 
                "critic_loss": critic_loss.item()}
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def save(self, path: str):
        """
        Save the agent's models.

        Args:
            path: Path to save the models
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }, path)
        
    def load(self, path: str):
        """
        Load the agent's models.

        Args:
            path: Path to load the models from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']