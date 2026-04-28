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


class MinFuelWrapper(gym.Wrapper):
    """
    Scenario 3 — Minimize Fuel (discrete MountainCar-v0).

    The "minimum-fuel" reward replaces the default ``-1 per step`` cost
    with a cost that is only incurred when the agent takes a non-null
    action (left or right). Idling (action=1) is free. Two scalar
    additions break two pathologies the bare cost has:

    Per-step terms:
        reward = -1                            if action != 1 (push)
        reward =  0                            if action == 1 (idle)
        reward += progress_shaping * height(p) every step (see below)
        reward += goal_bonus                   when the env terminates

    Goal bonus
    ----------
    Without a terminal bonus, the optimal policy is to idle forever:
    cumulative reward 0 strictly beats any goal-reaching trajectory's
    negative score. Adding a goal_bonus restores incentive to actually
    solve the task while keeping "cost is proportional to non-null
    actions" as the per-step objective.

    Energy shaping
    --------------
    Even with the goal bonus, vanilla DQN/PPO/A2C with the default
    ``gamma`` (~0.99) and short episodes (200 steps) cannot discover
    the goal during random exploration — MountainCar is a classic
    sparse-reward exploration benchmark. The bare wrapper produces
    0% success across all SB3 algorithms within reasonable training
    budgets, even with a generous goal bonus (the +100 reward never
    propagates back because no episode in the replay buffer ever
    reaches the goal during random exploration).

    We therefore stack the same energy-style shaping that Scenario 1
    uses (``EnergyShapingWrapper``):

        bonus = vel_scale * |velocity| + height_scale * height(position)

    where ``height(p) = sin(3p) * 0.45 + 0.55`` is the standard
    MountainCar altitude in [0, 1]. With ``vel_scale=100`` and
    ``height_scale=10``, the per-step bonus is at most ~17 — large
    enough that random pushes that happen to build velocity get
    immediate positive feedback. The -1 fuel cost is a soft signal
    that biases the policy toward idle when shaping reward is similar
    across actions.

    The fuel objective is preserved as a *soft* signal in this
    formulation; trained policies still differ from Scenario 1's
    energy-shaped agents because the fuel cost selects the more
    fuel-efficient resonance strategy when multiple goal-reaching
    policies score similarly on shaping.

    Setting ``progress_shaping=0`` recovers the unshaped wrapper, and
    ``goal_bonus=0`` recovers the pathological "do-nothing-wins"
    reward.
    """

    def __init__(self, env, goal_bonus=100.0, progress_shaping=1.0,
                 vel_scale=100.0, height_scale=10.0):
        super().__init__(env)
        self.goal_bonus = goal_bonus
        self.progress_shaping = progress_shaping
        self.vel_scale = vel_scale
        self.height_scale = height_scale

    @staticmethod
    def _height(pos):
        # Standard MountainCar altitude (matches Gymnasium's renderer).
        return float(np.sin(3.0 * pos) * 0.45 + 0.55)

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0 if int(action) == 1 else -1.0
        pos, vel = float(obs[0]), float(obs[1])
        if self.progress_shaping:
            reward += self.progress_shaping * (
                self.vel_scale * abs(vel)
                + self.height_scale * self._height(pos)
            )
        if terminated:
            reward += self.goal_bonus
        info = dict(info)
        info["raw_env_reward"] = float(_env_reward)
        info["non_null_action"] = (int(action) != 1)
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
