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
    Scenario 4 — Minimize Real Physical Time.

    Cost is proportional to the number of directional (non-null) actions taken.
    Equivalent to MinFuelWrapper in discrete action terms — both left and right
    actions incur cost; idling does not.

    This models the notion that the car's 'clock' only runs when the engine fires.

    reward = -1  if action in {0, 2} (left or right)
    reward =  0  if action == 1 (idle)
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = -1.0 if action in {0, 2} else 0.0
        return obs, reward, terminated, truncated, info


# Alias used in notebooks
CustomMountainCarWrapper = MinFuelWrapper
