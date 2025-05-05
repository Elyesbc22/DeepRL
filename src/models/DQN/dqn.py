import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any, Optional

class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)

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
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
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

class DQNAgent:
    """
    Deep Q-Network agent.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate at which epsilon decays
            buffer_size: Size of the replay buffer
            batch_size: Size of the batch for training
            target_update_freq: Frequency of target network updates
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Initialize Q-networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eval_mode: Whether to use evaluation mode (no exploration)
            
        Returns:
            Selected action
        """
        if (not eval_mode) and (random.random() < self.epsilon):
            # Exploration: select a random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: select the action with the highest Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self) -> Dict[str, float]:
        """
        Update the Q-network using a batch of experiences.
        
        Returns:
            Dictionary with loss information
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {"loss": loss.item()}
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
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
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        """
        Load the agent's models.
        
        Args:
            path: Path to load the models from
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
