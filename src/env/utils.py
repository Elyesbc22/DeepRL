import numpy as np
import random
from typing import List, Tuple, Optional, Union

def set_global_seeds(seed: int):
    """
    Set global seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
def normalize_observation(obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    """
    Normalize observations for stable training.
    
    Args:
        obs: Observation to normalize
        obs_mean: Mean of observations
        obs_std: Standard deviation of observations
        
    Returns:
        Normalized observation
    """
    return (obs - obs_mean) / (obs_std + 1e-8)

def running_mean_std(data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute running mean and standard deviation.
    
    Args:
        data: List of observations
        
    Returns:
        mean: Mean of observations
        std: Standard deviation of observations
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def collect_rollouts(env, policy, num_steps=10000):
    """
    Collect rollouts from an environment using a given policy.
    
    Args:
        env: Environment to collect rollouts from
        policy: Policy to use for actions
        num_steps: Number of steps to collect
        
    Returns:
        List of collected observations, actions, rewards, next_observations, dones
    """
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        observations.append(obs)
        
        action = policy(obs)
        actions.append(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        next_observations.append(next_obs)
        dones.append(done)
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    return observations, actions, rewards, next_observations, dones