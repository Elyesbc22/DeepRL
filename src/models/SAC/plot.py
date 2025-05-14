import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mean_and_std():
    # File paths
    file_paths = [
        "./log_train/log_train_s1/progress.txt",
        "./log_train/log_train_s2/progress.txt",
        "./log_train/log_train_s3/progress.txt"
    ]

    # Load the data
    dfs = [pd.read_csv(f, sep="\t") for f in file_paths]

    # Align all DataFrames on 'Epoch'
    min_epochs = min(len(df) for df in dfs)
    dfs = [df.iloc[:min_epochs] for df in dfs]

    # Stack data for aggregation
    stacked = pd.concat([df.set_index('Epoch') for df in dfs], axis=0, keys=range(3))

    # Compute mean and std for selected metrics
    metrics = ['AverageEpRet','AverageTestEpRet', 'LossPi', 'LossQ'] #AverageTestEpRet, AverageEpRet
    names_metric = ["train average reward", "test average reward", 'loss pi', 'loss q']
    mean = stacked.groupby('Epoch')[metrics].mean()
    std = stacked.groupby('Epoch')[metrics].std()

    # Plot
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Soft Actor Critic Training on CartPole-v1", fontsize=16)

    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        plt.plot(mean.index, mean[metric], label=f'Mean {metric}')
        plt.fill_between(mean.index, mean[metric] - std[metric], mean[metric] + std[metric], alpha=0.3)
        plt.title(f'{names_metric[i]} (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.show()


plot_mean_and_std()
