import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, Union

from src.env.environment import Environment

class DiscretizedActionWrapper(Environment):
    """
    Wrapper for environments with continuous action spaces to discretize them.
    This allows DQN (which requires discrete actions) to work with continuous environments.
    """
    def __init__(
        self, 
        env_name: str, 
        num_bins: int = 10,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            env_name: Name of the environment
            num_bins: Number of bins to discretize each action dimension
            seed: Random seed for reproducibility
            render_mode: Mode for rendering
        """
        super().__init__(env_name, seed, render_mode)
        
        # Only apply discretization if the action space is continuous
        if self.is_continuous:
            self.num_bins = num_bins
            self.action_low = self.action_space.low
            self.action_high = self.action_space.high
            
            # Create a discrete action space
            self.discrete_action_dim = num_bins ** self.action_space.shape[0]
            self.discrete_action_space = gym.spaces.Discrete(self.discrete_action_dim)
            
            # Store original action space
            self.original_action_space = self.action_space
            
            # Override action space
            self.action_space = self.discrete_action_space
            
            # Update is_continuous flag
            self.is_continuous = False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with a discretized action.
        
        Args:
            action: Discrete action index
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if hasattr(self, 'original_action_space'):
            # Convert discrete action to continuous
            continuous_action = self._discrete_to_continuous(action)
            return super().step(continuous_action)
        else:
            # If not continuous, just pass through
            return super().step(action)
    
    def _discrete_to_continuous(self, discrete_action: int) -> np.ndarray:
        """
        Convert a discrete action index to a continuous action vector.
        
        Args:
            discrete_action: Index of the discrete action
            
        Returns:
            Continuous action vector
        """
        # Convert discrete action index to multi-dimensional indices
        action_dim = self.original_action_space.shape[0]
        indices = []
        
        temp = discrete_action
        for _ in range(action_dim):
            indices.append(temp % self.num_bins)
            temp = temp // self.num_bins
        
        # Convert indices to continuous values
        continuous_action = np.zeros(action_dim)
        for i in range(action_dim):
            bin_size = (self.action_high[i] - self.action_low[i]) / self.num_bins
            continuous_action[i] = self.action_low[i] + (indices[i] + 0.5) * bin_size
        
        return continuous_action