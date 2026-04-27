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
    """

    def __init__(self, env, threshold=0.01):
        super().__init__(env)
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = -1.0 if abs(action[0]) > self.threshold else 0.0
        return obs, reward, terminated, truncated, info

# Alias used in notebooks
CustomMountainCarWrapper = MinFuelWrapper
