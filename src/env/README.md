# Importing an environment

```
from env import Environment, set_global_seeds

# Create an environment
env = Environment('CartPole-v1', seed=42)

# Use the environment
observation, info = env.reset()
```