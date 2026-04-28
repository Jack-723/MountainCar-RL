"""
Shared SB3 training helpers for the MountainCar group project.

Provides:
- make_ppo / make_a2c / make_dqn  -> SB3 model factories with hyperparameters
  taken from the SB3 RL Zoo MountainCar entries.
- train                            -> wraps model.learn() with eval + monitoring.
- save_metrics                     -> writes a metrics.json shard for cross-scenario merge.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@dataclass
class TrainResult:
    algo: str
    reward_mode: str
    total_timesteps: int
    seed: int
    eval_mean_reward: float
    eval_std_reward: float
    eval_mean_steps: float
    eval_success_rate: float
    model_path: str
    monitor_path: str
    extras: Dict[str, Any] = field(default_factory=dict)


def make_env(env_id: str, seed: int, monitor_path: Optional[str] = None,
             wrapper: Optional[Callable[[gym.Env], gym.Env]] = None) -> gym.Env:
    env = gym.make(env_id)
    if wrapper is not None:
        env = wrapper(env)
    env = Monitor(env, filename=monitor_path)
    env.reset(seed=seed)
    return env


def make_vec(env_id: str, n_envs: int, seed: int,
             monitor_dir: Optional[str] = None,
             wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
             normalize: bool = True):
    """Vectorised env factory for on-policy actor-critics (PPO/A2C).

    The RL-Zoo MountainCar configs assume vectorised rollouts; with a single
    env the on-policy gradient is too noisy to learn within reasonable budgets.

    NOTE: SB3's ``make_vec_env`` wraps Monitor *inside* the user wrapper, which
    means Monitor would log the raw env reward rather than the shaped reward
    actually used for training. We bypass that here by building each thunk
    manually so the wrap order is::

        gym.make(env_id) -> wrapper -> Monitor

    so that monitor CSVs reflect what the agent optimises.
    """
    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)

    def _make_thunk(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = gym.make(env_id)
            env.action_space.seed(seed + rank)
            if wrapper is not None:
                env = wrapper(env)
            monitor_path = (
                os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            )
            env = Monitor(env, filename=monitor_path)
            return env
        return _init

    vec = DummyVecEnv([_make_thunk(i) for i in range(n_envs)])
    vec.seed(seed)
    if normalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=0.99)
    return vec


def make_ppo(env, seed: int = 42, **overrides):
    """PPO with RL-Zoo MountainCar hyperparameters.

    Expects a *vectorised* env (use ``make_vec(..., n_envs=16)``) — the RL-Zoo
    config sets ``n_steps=16``, which gives a 256-sample rollout buffer with
    16 parallel envs.
    """
    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.98,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
    )
    kwargs.update(overrides)
    return PPO(**kwargs)


def make_a2c(env, seed: int = 42, **overrides):
    """A2C with RL-Zoo MountainCar hyperparameters (vectorised env expected)."""
    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.25,
        max_grad_norm=0.5,
        normalize_advantage=False,
        seed=seed,
        verbose=0,
    )
    kwargs.update(overrides)
    return A2C(**kwargs)


def make_dqn(env: gym.Env, seed: int = 42, **overrides):
    """DQN with RL-Zoo MountainCar hyperparameters."""
    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=4e-3,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=16,
        gradient_steps=8,
        target_update_interval=600,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        verbose=0,
    )
    kwargs.update(overrides)
    return DQN(**kwargs)


FACTORIES = {"ppo": make_ppo, "a2c": make_a2c, "dqn": make_dqn}


def make_ppo_continuous(env, seed: int = 42, **overrides):
    """PPO tuned for ``MountainCarContinuous-v0``.

    Differences from :func:`make_ppo` (the discrete-MountainCar config):

    * ``ent_coef=0.1`` and ``use_sde=True`` — the dense ``-0.1·a²`` cost in
      the default continuous env (and the per-action cost in Scenario 4)
      makes "output ~zero force" a strong local optimum; both knobs push
      the policy to keep exploring forces away from zero.
    * Larger ``n_steps`` (2048 / 256 / 64) — continuous control benefits
      from longer rollouts. Defaults here assume a *single* env (as used
      in Scenario 2 / Scenario 4) but vectorisation works too.
    """
    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        seed=seed,
        verbose=0,
    )
    kwargs.update(overrides)
    return PPO(**kwargs)


def make_sac(env, seed: int = 42, **overrides):
    """SAC tuned for ``MountainCarContinuous-v0`` (Scenario 2 / 4).

    Hyperparameters are taken from the SB3 RL Zoo entry for
    ``MountainCarContinuous-v0`` and are critical for the
    sparse-bonus + per-step-cost reward structure:

    * ``gamma=0.9999`` — the goal bonus only fires at terminal step,
      so we need a very-long-horizon discount to back-propagate it
      through the trajectory.
    * ``log_std_init=-3`` — small initial action stds keep the agent
      near the cost-free zone early on while SDE provides exploration.
    * ``net_arch=[400, 300]`` — TD3-style wider critic.
    * ``use_sde=True`` with ``sde_sample_freq=4`` — gives state-dependent
      exploration noise that the entropy bonus alone struggles to match
      on this env.
    """
    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=7.3e-4,
        buffer_size=50_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.9999,
        tau=0.02,
        train_freq=8,
        gradient_steps=8,
        learning_starts=10_000,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
        seed=seed,
        verbose=0,
    )
    kwargs.update(overrides)
    return SAC(**kwargs)


FACTORIES_CONTINUOUS = {"ppo": make_ppo_continuous, "sac": make_sac}


def train(model, total_timesteps: int, eval_env: Optional[gym.Env] = None,
          eval_freq: int = 5_000, n_eval_episodes: int = 10,
          log_dir: Optional[str] = None, progress_bar: bool = False):
    """Train a model with optional periodic evaluation."""
    callback = None
    if eval_env is not None and log_dir is not None:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best"),
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=progress_bar)
    return model


def evaluate(model, env, n_episodes: int = 100, seed: int = 42,
             max_steps: int = 200) -> Dict[str, float]:
    """Deterministic evaluation. Reports reward, steps, success rate.

    Accepts either a single Gymnasium env (for DQN single-env training) or a
    SB3 ``VecEnv``/``VecNormalize`` (for PPO/A2C trained with normalisation).
    For VecEnvs we drive only the first sub-env (n_envs=1 expected at eval).
    """
    is_vec = hasattr(env, "num_envs")
    rewards, lengths, successes = [], [], []
    for ep in range(n_episodes):
        if is_vec:
            obs = env.reset()
            done, total_r, steps, terminated = False, 0.0, 0, False
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, dones, infos = env.step(action)
                total_r += float(r[0])
                steps += 1
                done = bool(dones[0])
                if done:
                    info = infos[0]
                    # Truncated by TimeLimit -> not a real terminal goal-reach
                    truncated = info.get("TimeLimit.truncated", False)
                    terminated = (not truncated) and steps < max_steps
        else:
            obs, _ = env.reset(seed=seed + ep)
            done, total_r, steps, terminated = False, 0.0, 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                total_r += float(r)
                steps += 1
                done = term or trunc
                terminated = bool(term)
                if steps >= max_steps:
                    break
        rewards.append(total_r)
        lengths.append(steps)
        successes.append(int(terminated))
    return dict(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_steps=float(np.mean(lengths)),
        std_steps=float(np.std(lengths)),
        success_rate=float(np.mean(successes)),
        n_episodes=int(n_episodes),
    )


def make_eval_vec(env_id: str, vecnorm_path: str,
                  monitor_path: Optional[str] = None) -> VecNormalize:
    """Build a single-env VecNormalize loaded from saved training stats.

    Use this to evaluate a PPO/A2C model that was trained with VecNormalize:
    the policy expects normalised observations, so eval must apply the same
    transformation. We disable training and reward normalisation so that
    rewards reported back are the raw env rewards.
    """
    def _mk():
        e = gym.make(env_id)
        if monitor_path is not None:
            e = Monitor(e, filename=monitor_path)
        return e
    vec = DummyVecEnv([_mk])
    vec = VecNormalize.load(vecnorm_path, vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def save_metrics(results_dir: str, scenario: int, results: Dict[str, TrainResult]) -> str:
    """Write a single metrics.json with one entry per (algo, reward_mode)."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "metrics.json")
    payload = {
        "scenario": scenario,
        "env_id": "MountainCar-v0",
        "results": {key: asdict(val) for key, val in results.items()},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path
