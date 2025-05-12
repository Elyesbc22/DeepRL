import matplotlib.pyplot as plt

def plot_rewards(rewards, title="Training Rewards", save_path=None):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()