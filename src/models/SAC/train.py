import gym
# Option 1 – pull the callable out of the module file directly
from spinningup.spinup.algos.pytorch.sac.sac import sac          # <-- the function
from spinningup.spinup.algos.pytorch.sac import core             # <-- the core nets


# --- environment factory ---------------------------------------------------
def env_fn():
    return gym.make("MountainCarContinuous-v0")

# --- training --------------------------------------------------------------
if __name__ == "__main__":
    sac(env_fn,
        actor_critic = core.MLPActorCritic,          # use the built‑in nets
        ac_kwargs    = dict(hidden_sizes=[256, 256]),
        steps_per_epoch = 10_000,
        epochs          = 300,
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
        seed              = 1,
        save_freq         = 10,
    )
