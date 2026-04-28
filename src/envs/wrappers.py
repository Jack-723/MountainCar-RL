import gymnasium as gym
import numpy as np


class EnergyShapingWrapper(gym.Wrapper):
    """
    Energy-style reward shaping for MountainCar (Scenario 1).

    Adds two non-negative bonuses to the per-step reward:

    * ``vel_scale * |velocity|`` — encourages swinging (useful early signal).
    * ``height_scale * height(position)`` — encourages climbing
      (breaks the "swing forever in the valley" local optimum that velocity-only
      shaping creates).

    This is *not* potential-based (Ng, Harada & Russell 1999), so it can in
    principle bias the optimal policy — but the bonus is monotone in altitude,
    so the qualitative shaped optimum is still "reach the flag". We always
    evaluate against the unshaped env so reported numbers correspond to
    Scenario 1's objective.

    ``scale=0.0`` recovers the unmodified reward.
    """

    def __init__(self, env, scale=1.0, vel_scale=100.0,
                 height_scale=10.0, goal_bonus=100.0, **_):
        super().__init__(env)
        # ``scale`` is a global multiplier kept for backward compat.
        self.vel_scale = scale * vel_scale
        self.height_scale = scale * height_scale
        self.goal_bonus = scale * goal_bonus

    @staticmethod
    def _height(pos):
        # Same formula the Gymnasium MountainCar visualization uses.
        return float(np.sin(3.0 * pos) * 0.45 + 0.55)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos, vel = float(obs[0]), float(obs[1])
        bonus = self.vel_scale * abs(vel) + self.height_scale * self._height(pos)
        if terminated:
            bonus += self.goal_bonus
        info = dict(info)
        info["raw_reward"] = float(reward)
        return obs, reward + bonus, terminated, truncated, info


class MinFuelWrapper(gym.RewardWrapper):
    """
    Scenario 3 — Minimize Fuel.

    Replaces the default reward (-1 per step) with a cost that is only
    incurred when the agent takes a non-null action (left or right).
    Idling (action=1) is free.

    reward = -1  if action != 1 (i.e. agent pushed left or right)
    reward =  0  if action == 1 (idle)
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0 if action == 1 else -1.0
        return obs, reward, terminated, truncated, info


class MinRealTimeWrapper(gym.RewardWrapper):
    """
    Scenario 4 — Continuous, Minimum Steps.

    Cost is linearly proportional to the number of non-null actions taken.
    A non-null action is any action where |force| > threshold.

        reward = -1            if |action| > threshold (non-null action)
        reward =  0            if |action| <= threshold (idle)
        reward += goal_bonus   when the env terminates (position >= 0.45)

    Why the goal bonus is required
    ------------------------------
    Without a terminal bonus, the optimal policy under this cost is to
    output |action| <= threshold forever and never terminate (cumulative
    reward = 0 dominates any goal-reaching trajectory's negative score).
    The bonus restores incentive to actually solve the task while keeping
    "cost is proportional to non-null actions" as the per-step objective.

    Setting ``goal_bonus=0`` recovers the pathological wrapper-only reward
    so the do-nothing exploit can be demonstrated in the notebook.

    The wrapper also writes the absolute action magnitude and a
    ``non_null_action`` flag into ``info`` so evaluation code can compute
    fuel statistics directly without re-running the agent.
    """

    def __init__(self, env, threshold=0.01, goal_bonus=100.0):
        super().__init__(env)
        self.threshold = threshold
        self.goal_bonus = goal_bonus

    def step(self, action):
        # action arrives as a 1-D ndarray for MountainCarContinuous-v0;
        # be defensive in case a scalar is passed for unit testing.
        a = float(action[0]) if hasattr(action, "__len__") else float(action)
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        is_null = abs(a) <= self.threshold
        reward = 0.0 if is_null else -1.0
        if terminated:
            reward += self.goal_bonus
        info = dict(info)
        info["raw_env_reward"] = float(_env_reward)
        info["abs_action"] = abs(a)
        info["non_null_action"] = (not is_null)
        return obs, reward, terminated, truncated, info

# Alias used in notebooks
CustomMountainCarWrapper = MinFuelWrapper
