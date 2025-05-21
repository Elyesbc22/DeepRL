# ---------------------------------------------------------------
# PPO implementation (fixed v2)
# ---------------------------------------------------------------
# What changed in v2 (compared to the previous canvas version):
#   • Kept all earlier fixes **plus**
#   • Added a `gae_lambda` alias parameter so external training scripts that
#     still pass `gae_lambda=0.95` don't error out. `lam` and `gae_lambda`
#     now map to the same internal variable; if both are given, `gae_lambda`
#     wins.
# ---------------------------------------------------------------

from __future__ import annotations

import os, sys
from typing import Dict, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Assume repo layout unchanged
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env.environment import Environment
from env.utils import set_global_seeds

# ---------------------------------------------------------------------
# Running mean / std helper
# ---------------------------------------------------------------------
class RunningMeanStd:
    """Tracks running mean & variance (Welford)."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var  = np.ones(shape,  np.float32)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        x        = np.asarray(x, np.float32)
        b_mean   = x.mean(axis=0)
        b_var    = x.var(axis=0)
        b_count  = x.shape[0]
        delta    = b_mean - self.mean
        tot_cnt  = self.count + b_count
        new_mean = self.mean + delta * b_count / tot_cnt
        m_a      = self.var * self.count
        m_b      = b_var * b_count
        M2       = m_a + m_b + np.square(delta) * self.count * b_count / tot_cnt
        new_var  = M2 / tot_cnt
        self.mean, self.var, self.count = new_mean, new_var, tot_cnt

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# ---------------------------------------------------------------------
# Actor‑Critic network
# ---------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 continuous: bool = False, action_std_init: float = 0.6):
        super().__init__()
        self.continuous = continuous
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        if continuous:
            self.register_buffer("action_var", torch.full((action_dim,), action_std_init ** 2))
        # Orthogonal init
        def ortho(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        self.apply(ortho)

    def forward(self, x: torch.Tensor):
        logits = self.actor(x)
        value  = self.critic(x)
        return logits, value

    def act(self, x: torch.Tensor):
        logits, value = self(x)
        if self.continuous:
            dist = Normal(logits, torch.sqrt(self.action_var))
            action = dist.sample(); logp = dist.log_prob(action).sum(-1)
            return action.squeeze(0).cpu().numpy(), logp, value
        probs = torch.softmax(logits, -1); dist = Categorical(probs)
        action = dist.sample(); return action.item(), dist.log_prob(action), value

    def evaluate(self, x: torch.Tensor, a: torch.Tensor):
        logits, value = self(x)
        if self.continuous:
            dist = Normal(logits, torch.sqrt(self.action_var))
            logp = dist.log_prob(a).sum(-1); ent = dist.entropy().sum(-1).mean()
            return logp, value, ent
        probs = torch.softmax(logits, -1); dist = Categorical(probs)
        return dist.log_prob(a), value, dist.entropy().mean()

    def set_action_std(self, std: float):
        if self.continuous:
            with torch.no_grad(): self.action_var.fill_(std ** 2)

# ---------------------------------------------------------------------
# Buffer (stores normalised states + scaled rewards)
# ---------------------------------------------------------------------
class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.values, self.logps, self.dones = [], [], [], [], [], []

    def add(self, s, a, r, v, lp, d):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.values.append(v.item()); self.logps.append(lp.item()); self.dones.append(d)

    def compute_returns_adv(self, last_v, ret_rms: Optional[RunningMeanStd], gamma=0.99, lam=0.95):
        vals = self.values + [last_v]
        adv, ret = np.zeros(len(self.rewards), np.float32), np.zeros_like(self.rewards, np.float32)
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            next_v = vals[t + 1] * (1. - self.dones[t])
            delta  = self.rewards[t] + gamma * next_v - vals[t]
            gae    = delta + gamma * lam * (1. - self.dones[t]) * gae
            adv[t] = gae; ret[t] = adv[t] + vals[t]
        if ret_rms is not None:
            ret_rms.update(ret); ret = ret / (np.sqrt(ret_rms.var) + 1e-8)
        self.advantages, self.returns = adv, ret

    def get(self):
        s = np.array(self.states, np.float32)
        a = np.array(self.actions, np.float32 if isinstance(self.actions[0], np.ndarray) else np.int64)
        return s, a, np.array(self.logps, np.float32), self.advantages, self.returns, np.array(self.values, np.float32)

    def clear(self):
        self.__init__()

# ---------------------------------------------------------------------
# PPO Agent (now accepting both `lam` and `gae_lambda`)
# ---------------------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, *, lr=3e-4, gamma=0.99,
                 lam: Optional[float] = None, gae_lambda: Optional[float] = None,
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 continuous=False, action_std=0.6, device=None,
                 total_steps=1_000_000):
        # allow both names; gae_lambda overrides lam if both supplied
        self.lam = gae_lambda if gae_lambda is not None else (lam if lam is not None else 0.95)
        self.gamma, self.clip = gamma, clip_ratio
        self.v_coef, self.ent_init, self.max_grad = value_coef, entropy_coef, max_grad_norm
        self.total_steps_budget, self.step_count = total_steps, 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim, continuous, action_std).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, lambda e: 1. - min(e * 0.01, 0.9))
        self.buf = PPOBuffer()
        self.obs_rms, self.rew_rms, self.ret_rms = RunningMeanStd((state_dim,)), RunningMeanStd(()), RunningMeanStd(())
        self.continuous = continuous

    # -------------------------------------------------------------
    # Data collection helpers
    # -------------------------------------------------------------
    def select_action(self, state: np.ndarray):
        norm_s = self.obs_rms.normalize(state)
        s_t = torch.tensor(norm_s, dtype=torch.float32).unsqueeze(0).to(self.device)
        a, _, _ = self.policy.act(s_t)
        return a

    def collect(self, env: Environment, horizon=2048):
        s, _ = env.reset()
        if hasattr(env.observation_space, "low"):
            s = np.clip(s, env.observation_space.low, env.observation_space.high)
        s = s.astype(np.float32).flatten(); self.obs_rms.update([s])
        ep_rew, ep_len, eps, tot_rew = 0., 0, 0, 0.
        done = False
        for _ in range(horizon):
            self.step_count += 1
            norm_s = self.obs_rms.normalize(s)
            a, lp, v = self.policy.act(torch.tensor(norm_s).unsqueeze(0).to(self.device))
            n_s, r, term, trunc, _ = env.step(a)
            self.rew_rms.update([r]); r_s = r / (np.sqrt(self.rew_rms.var) + 1e-8)
            if hasattr(env.observation_space, "low"):
                n_s = np.clip(n_s, env.observation_space.low, env.observation_space.high)
            n_s = n_s.astype(np.float32).flatten(); self.obs_rms.update([n_s])
            done = term or trunc
            self.buf.add(norm_s, a, r_s, v, lp, done)
            ep_rew += r; ep_len += 1; s = n_s
            if done:
                eps += 1; tot_rew += ep_rew
                s, _ = env.reset();
                if hasattr(env.observation_space, "low"):
                    s = np.clip(s, env.observation_space.low, env.observation_space.high)
                s = s.astype(np.float32).flatten(); self.obs_rms.update([s])
                ep_rew, ep_len = 0., 0
        # bootstrap value
        last_v = 0.
        if not done:
            last_v = self.policy.forward(torch.tensor(self.obs_rms.normalize(s)).unsqueeze(0).
                                         to(self.device))[1].item()
        self.buf.compute_returns_adv(last_v, self.ret_rms, self.gamma, self.lam)
        return {"episodes": eps, "avg_reward": tot_rew / max(eps, 1)}

    def collect_trajectory(self, env: Environment, horizon=2048):
        """Alias for collect() method for backward compatibility."""
        return self.collect(env, horizon)

    # -------------------------------------------------------------
    # PPO update
    # -------------------------------------------------------------
    def update(self, epochs=4, batch=256):
        s, a, old_lp, adv, ret, old_v = self.buf.get()
        s      = torch.tensor(s, dtype=torch.float32).to(self.device)
        if self.continuous:
            a = torch.tensor(a, dtype=torch.float32).to(self.device)
        else:
            a = torch.tensor(a, dtype=torch.long).to(self.device)
        old_lp = torch.tensor(old_lp).to(self.device)
        adv    = torch.tensor(adv).to(self.device)
        ret    = torch.tensor(ret).to(self.device)
        old_v  = torch.tensor(old_v).to(self.device)
        adv    = (adv - adv.mean()) / (adv.std() + 1e-8)

        ent_coef = max(self.ent_init * (1. - self.step_count / self.total_steps_budget), 0.001)
        pol_loss_tot = v_loss_tot = ent_tot = 0.
        for _ in range(epochs):
            idx = np.random.permutation(len(s))
            for st in range(0, len(s), batch):
                b = idx[st: st + batch]
                lp, v, ent = self.policy.evaluate(s[b], a[b])
                ratio = torch.exp(lp - old_lp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1.-self.clip, 1.+self.clip) * adv[b]
                pol_loss = -torch.where(adv[b] >= 0, torch.min(s1, s2), torch.max(s1, s2)).mean()
                v_pred_clip = old_v[b] + torch.clamp(v.squeeze() - old_v[b], -self.clip, self.clip)
                v_loss = 0.5 * torch.max((v.squeeze()-ret[b])**2, (v_pred_clip-ret[b])**2).mean()
                loss = pol_loss + self.v_coef * v_loss - ent_coef * ent
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.opt.step()
                pol_loss_tot += pol_loss.item(); v_loss_tot += v_loss.item(); ent_tot += ent.item()
        self.scheduler.step(); self.buf.clear()
        n_up = epochs * ((len(s)+batch-1)//batch)
        return {"policy_loss": pol_loss_tot/n_up, "value_loss": v_loss_tot/n_up, "entropy": ent_tot/n_up}

    # -------------------------------------------------------------
    # Checkpoint helpers (fix for AttributeError)
    # -------------------------------------------------------------
    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(),
                    "opt": self.opt.state_dict(),
                    "sched": self.scheduler.state_dict(),
                    "obs_mean": self.obs_rms.mean, "obs_var": self.obs_rms.var, "obs_count": self.obs_rms.count,
                    "steps": self.step_count}, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(ckpt["policy"])
        self.opt.load_state_dict(ckpt["opt"])
        self.scheduler.load_state_dict(ckpt["sched"])
        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = ckpt["obs_mean"], ckpt["obs_var"], ckpt["obs_count"]
        self.step_count = ckpt.get("steps", 0)

# ---------------------------------------------------------------------
# Utility to set global seeds (unchanged helper)
# ---------------------------------------------------------------------
__all__ = ["PPOAgent", "RunningMeanStd", "ActorCritic"]
