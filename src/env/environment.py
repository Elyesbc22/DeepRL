import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union

class Environment:
    """
    Wrapper class for OpenAI Gym environments used in the RL course.
    Supports Cartpole, MountainCar, MountainCarContinuous, Acrobot, and Pendulum.
    """
    def __init__(
        self, 
        env_name: str, 
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize an environment.
        
        Args:
            env_name: Name of the environment ('CartPole-v1', 'MountainCar-v0', 
                    'MountainCarContinuous-v0', 'Acrobot-v1', 'Pendulum-v1')
            seed: Random seed for reproducibility
            render_mode: Mode for rendering ('human', 'rgb_array', etc.)
        """
        self.env_name = env_name
        self.seed = seed
        
        self.env = gym.make(env_name, render_mode=render_mode)
        
        if seed is not None:
            self.env.reset(seed=seed)
            np.random.seed(seed)
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.is_continuous = isinstance(self.action_space, gym.spaces.Box)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        return self.env.reset()
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        return self.env.step(action)
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def state_dim(self) -> int:
        """Get the dimension of the observation space."""
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return self.observation_space.n
        return self.observation_space.shape[0]
    
    @property
    def action_dim(self) -> int:
        """Get the dimension of the action space."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n
        return self.action_space.shape[0]
    
    @property
    def action_bound(self) -> Optional[float]:
        """Get the bound of actions for continuous action spaces."""
        if self.is_continuous:
            return self.action_space.high[0]
        return None

    @staticmethod
    def get_env_list():
        """
        Returns a list of supported environments.
        """
        return [
            'CartPole-v1',
            'MountainCar-v0',
            'MountainCarContinuous-v0',
            'Acrobot-v1',
            'Pendulum-v1'
        ]