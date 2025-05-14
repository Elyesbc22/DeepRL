import gym
import numpy as np
from gym import spaces

class DiscreteToContinuousActionWrapper(gym.Env):
    """
    Wrap a discrete-action environment to present a fake continuous action space.
    This allows using continuous-action algorithms like SAC with discrete environments.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete), \
            "Environment must have a discrete action space"

        # Fake continuous action space in range [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # Convert continuous value in [-1, 1] to discrete index
        # E.g., for 2 actions: < 0 -> 0, >= 0 -> 1
        if isinstance(action, np.ndarray):
            action = action[0]
        discrete_action = int((action > 0.0))  # Only for 2 discrete actions
        return self.env.step(discrete_action)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()
