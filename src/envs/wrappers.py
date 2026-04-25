import gymnasium as gym
import numpy as np


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
