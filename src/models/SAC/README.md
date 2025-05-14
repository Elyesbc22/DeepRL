# DeepRL → Soft Actor-Critic

Soft Actor-Critic (SAC) is a deep reinforcement learning algorithm that blends off-policy learning with a maximum entropy framework. This design promotes both efficient exploration and stable learning by encouraging the agent to act as randomly as possible while still maximizing expected rewards.

Although SAC is originally designed for continuous action spaces, in this implementation it has been discretized to handle two actions: **left** or **right**.

![Alt text](./SAC_CARTPOLE.png "Soft Actor Critic Training on CartPole-v1")

As shown in the plot above, after approximately 80 training epochs, the model consistently achieves the maximum reward of **500**, indicating that it has successfully learned to solve the CartPole-v1 task.

## How to Train and Plot SAC on CartPole-v1

<ol>
  <li>Install required packages from `requirements.txt`.</li>
  <li>Install the [Spinning Up](https://spinningup.openai.com/en/latest/user/installation.html) library.</li>
  <li>run python main.py from the SAC folder</li>
  <li>Update the data path in plot.py to point to the new training log. The correct path will be displayed at the start of the console output during training.</li>
  <li>run python plot.py</li>
</ol>