import gym
# Option 1 – pull the callable out of the module file directly
from spinup.algos.pytorch.sac.sac import sac          # <-- the function
from spinup.algos.pytorch.sac import core             # <-- the core nets
from wrapper_continuous_to_disrete_env import DiscreteToContinuousActionWrapper
from spinup.utils.run_utils import setup_logger_kwargs


# --- environment factory ---------------------------------------------------
def env_fn():
    base_env = gym.make("CartPole-v1")
    return DiscreteToContinuousActionWrapper(base_env)

# --- training --------------------------------------------------------------
if __name__ == "__main__":
    seed = 3
    sac(env_fn,
        actor_critic = core.MLPActorCritic,          # use the built‑in nets
        ac_kwargs    = dict(hidden_sizes=[256, 256]),
        steps_per_epoch = 2_000,
        epochs          = 100,
        gamma           = 0.98,
        polyak          = 0.995,
        lr              = 2e-4,
        alpha           = 0.2,
        batch_size      = 64,
        start_steps     = 10_000,
        update_after    = 1_000,
        update_every    = 1,
        num_test_episodes = 10,
        max_ep_len        = 999,
        seed              = seed,
        save_freq         = 10,
        logger_kwargs     = setup_logger_kwargs(exp_name="log_train", seed=seed)
    )
