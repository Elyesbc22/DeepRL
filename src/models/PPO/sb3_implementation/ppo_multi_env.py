#!/usr/bin/env python3
"""
Train PPO on multiple Gymnasium environments using hyperparameters
from ppo.yml and plot average training and evaluation curves across seeds.
"""
import os
import yaml
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Helper functions ---
def linear_schedule(initial_value: float):
    """
    Return a function that linearly decays from initial_value to 0.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def load_hyperparams(yaml_path: str):
    """
    Load hyperparameters from a YAML file.
    Returns a dict: env_id -> (policy_str, params_dict)
    """
    import yaml
    import os
    
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    parsed = {}
    for env_id, params in raw.items():
        p = params.copy()
        p.pop("n_timesteps", None)
        p.pop("n_envs", None)
        policy = p.pop("policy", "MlpPolicy")
        p.pop("normalize", None)
        
        # Handle policy_kwargs which could be a string with dict() and nn.ReLU
        if "policy_kwargs" in p and isinstance(p["policy_kwargs"], str):
            try:
                # First try ast.literal_eval (safe but limited)
                import ast
                try:
                    evaluated_kwargs = ast.literal_eval(p["policy_kwargs"])
                    if isinstance(evaluated_kwargs, dict):
                        p["policy_kwargs"] = evaluated_kwargs
                    else:
                        raise ValueError("Not a dictionary")
                except (ValueError, SyntaxError):
                    # If that fails, use restricted eval for dict() and nn.X
                    if "dict(" in p["policy_kwargs"]:
                        # Import torch.nn for nn.X references
                        import torch.nn as nn
                        
                        # Create a safe environment with limited access
                        safe_globals = {"__builtins__": {}}
                        safe_locals = {"dict": dict, "nn": nn}
                        
                        # Evaluate the expression with restricted globals
                        p["policy_kwargs"] = eval(p["policy_kwargs"], safe_globals, safe_locals)
                    else:
                        raise ValueError(f"Unrecognized policy_kwargs format: {p['policy_kwargs']}")
            except Exception as e:
                raise ValueError(
                    f"Error parsing policy_kwargs for {env_id}: '{p['policy_kwargs']}'. "
                    f"Original error: {e}"
                )
        
        # Parse schedules
        for key in ("learning_rate", "clip_range"):
            if key in p and isinstance(p[key], str):
                prefix, num = p[key].split("_", 1)
                num = float(num)
                if prefix == "lin":
                    p[key] = linear_schedule(num)
                elif prefix == "const":
                    p[key] = num
                else:
                    raise ValueError(f"Unknown schedule prefix {prefix} in {p[key]}")
        
        parsed[env_id] = (policy, p)
    return parsed


# --- Callbacks ---
class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar during training.
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", position=2, leave=False)

    def _on_step(self) -> bool:
        if self.pbar:
            # update by number of calls since last update
            self.pbar.update(self.n_calls - self.pbar.n)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None


class VerboseEvalCallback(EvalCallback):
    """
    Enhanced EvalCallback that prints evaluation results.
    """
    def _on_step(self) -> bool:
        cont = super()._on_step()
        # Print only at evaluation steps
        if self.eval_freq and self.n_calls % self.eval_freq == 0 and hasattr(self, 'last_mean_reward'):
            print(f"    → Step {self.num_timesteps}: Eval mean={self.last_mean_reward:.2f}")
        return cont


def create_monitored_env(env_id_local, monitor_csv_path=None):
    env = gym.make(env_id_local)
    if monitor_csv_path:
        return Monitor(env, monitor_csv_path)
    else:
        return Monitor(env, filename=None)


# --- Main script ---
if __name__ == "__main__":
    # Configuration
    results_dir = "ppo_eval_logs"
    yaml_path = "ppo.yml"
    env_ids = [
        #"CartPole-v1",
        #"MountainCar-v0",
        "MountainCarContinuous-v0",
        "Acrobot-v1",
        "Pendulum-v1",
    ]
    seeds = [0, 1, 2]
    total_timesteps = 200_000
    eval_freq = 10_000
    n_eval_episodes = 5

    # Load hyperparameters
    hyperparams = load_hyperparams(yaml_path)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loaded hyperparameters from {yaml_path}:")
    for env, (pol, hp) in hyperparams.items():
        print(f"  - {env}: policy={pol}, params_keys={list(hp.keys())}")

    # Train & evaluate
    for env_idx, env_id in enumerate(env_ids, start=1):
        if env_id not in hyperparams:
            raise KeyError(f"No hyperparameters found for {env_id} in {yaml_path}")
        policy_class, current_hyperparams = hyperparams[env_id]
        print(f"\n=== [{env_idx}/{len(env_ids)}] Training on {env_id} with policy {policy_class}")

        for seed_idx, seed_val in enumerate(seeds, start=1):
            run_id = f"{env_id.replace('-', '_')}_seed{seed_val}"
            log_path = os.path.join(results_dir, run_id)
            os.makedirs(log_path, exist_ok=True)
            print(f"  ► Seed {seed_idx}/{len(seeds)} = {seed_val} → logs/{run_id}")

            # Create environments
            train_csv = os.path.join(log_path, "training_monitor.csv")
            train_env = DummyVecEnv([lambda eid=env_id, p=train_csv: create_monitored_env(eid, p)])
            eval_env  = DummyVecEnv([lambda eid=env_id: create_monitored_env(eid, None)])

            # Callbacks
            eval_cb     = VerboseEvalCallback(
                eval_env,
                best_model_save_path=None,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=0,
            )
            progress_cb = ProgressBarCallback(total_timesteps)

            # Initialize and train
            print("    • Initializing PPO model …")
            model = PPO(
                policy=policy_class,
                env=train_env,
                seed=seed_val,
                verbose=0,
                **current_hyperparams
            )
            print(f"    • Training for {total_timesteps} timesteps …")
            start_time = time.time()
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_cb, progress_cb]
            )
            print(f"    • Done in {time.time() - start_time:.1f}s")

    # Aggregate and plot results
    print("\nAll training complete. Aggregating results and plotting …")
    plt.figure(figsize=(14, 8))
    plot_timesteps = np.linspace(0, total_timesteps, num=200)

    for env_id in env_ids:
        eval_series = []
        train_series = []
        print(f"Processing {env_id} …")
        for seed_val in seeds:
            run_id = f"{env_id.replace('-', '_')}_seed{seed_val}"
            path = os.path.join(results_dir, run_id)
            # Eval data
            data = np.load(os.path.join(path, "evaluations.npz"))
            mean_eval = data["results"].mean(axis=1)
            steps_eval = data["timesteps"]
            eval_series.append(pd.Series(mean_eval, index=steps_eval))
            # Train data
            df_train = pd.read_csv(os.path.join(path, "training_monitor.csv"), skiprows=1)
            df_train['timesteps'] = df_train['l'].cumsum()
            rewards = df_train['r'].values
            steps_train = df_train['timesteps'].values
            interp = np.interp(plot_timesteps, steps_train, rewards, left=np.nan, right=np.nan)
            train_series.append(pd.Series(interp, index=plot_timesteps))

        # Plot eval
        df_eval = pd.concat(eval_series, axis=1)
        m_eval = df_eval.mean(axis=1)
        s_eval = df_eval.std(axis=1)
        plt.plot(m_eval.index, m_eval.values, marker='o', label=f"{env_id} (Eval)")
        plt.fill_between(m_eval.index, m_eval - s_eval, m_eval + s_eval, alpha=0.2)
        # Plot train
        df_train_all = pd.concat(train_series, axis=1)
        m_train = df_train_all.mean(axis=1)
        s_train = df_train_all.std(axis=1)
        plt.plot(m_train.index, m_train.values, linestyle='--', label=f"{env_id} (Train)")
        plt.fill_between(m_train.index, m_train - s_train, m_train + s_train, alpha=0.1)

    plt.xlabel("Timesteps")
    plt.ylabel(f"Average Reward\n(Eval over {n_eval_episodes} eps; Train interpolated)")
    plt.title(f"PPO Learning Curves - {total_timesteps} Steps")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    out_file = "ppo_multi_env_learning_curves_with_yml.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved as {out_file}")
    plt.show()
