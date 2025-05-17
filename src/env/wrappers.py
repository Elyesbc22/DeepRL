import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, Union

from . import Environment

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

class RewardVelocityModifierWrapper(Environment):
    """
    Wrapper for environments to modify the reward function.
    This allows adding a velocity component to the reward to encourage specific behaviors.
    """
    def __init__(
        self, 
        env_name: str, 
        velocity_coefficient: float = 0.2,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the wrapper.

        Args:
            env_name: Name of the environment
            velocity_coefficient: Coefficient to scale the velocity component added to the reward
            seed: Random seed for reproducibility
            render_mode: Mode for rendering
        """
        super().__init__(env_name, seed, render_mode)
        self.velocity_coefficient = velocity_coefficient

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment and modify the reward.

        Args:
            action: Action to take in the environment

        Returns:
            observation: Next observation
            reward: Modified reward (original reward + velocity * coefficient)
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Handle discretized actions for continuous environments
        if hasattr(self, '_discrete_to_continuous') and isinstance(action, int):
            # Convert discrete action to continuous
            action = self._discrete_to_continuous(action)

        observation, reward, terminated, truncated, info = super().step(action)

        # For MountainCar and MountainCarContinuous, velocity is the second component of the observation
        # For other environments, this might need to be adjusted
        if 'MountainCar' in self.env_name and len(observation) > 1:
            velocity = observation[1]
            modified_reward = reward + velocity * self.velocity_coefficient
            return observation, modified_reward, terminated, truncated, info

        return observation, reward, terminated, truncated, info


class ContinuousActionWrapper(Environment):
    """
    Wrapper for environments with discrete action spaces to make them continuous.
    This allows algorithms like TD3 (which require continuous actions) to work with discrete environments.
    """
    def __init__(
        self, 
        env_name: str, 
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the wrapper.

        Args:
            env_name: Name of the environment
            seed: Random seed for reproducibility
            render_mode: Mode for rendering
        """
        super().__init__(env_name, seed, render_mode)

        # Only apply conversion if the action space is discrete
        if not self.is_continuous:
            # Store original discrete action space
            self.original_action_space = self.action_space
            self.discrete_action_dim = self.action_space.n

            # Create a continuous action space
            # Use a Box with dimensions [0, 1] for each discrete action
            self.continuous_action_space = gym.spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(1,),  # Single dimension for simplicity
                dtype=np.float32
            )

            # Override action space
            self.action_space = self.continuous_action_space

            # Update is_continuous flag
            self.is_continuous = True

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with a continuous action.

        Args:
            action: Continuous action vector

        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if hasattr(self, 'original_action_space'):
            # Convert continuous action to discrete
            discrete_action = self._continuous_to_discrete(action)
            return super().step(discrete_action)
        else:
            # If not discrete, just pass through
            return super().step(action)

    def _continuous_to_discrete(self, continuous_action: np.ndarray) -> int:
        """
        Convert a continuous action vector to a discrete action index.

        Args:
            continuous_action: Continuous action vector

        Returns:
            Discrete action index
        """
        # Scale the continuous action to the range [0, discrete_action_dim)
        # and convert to an integer
            # Ensure it's in [0, 1), or map 1.0 exactly to last bin safely
        clipped_action = np.clip(continuous_action[0], 0.0, 1.0 - 1e-8)
        discrete_action = int(clipped_action * self.discrete_action_dim)
        return discrete_action
