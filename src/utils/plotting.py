import numpy as np
import matplotlib.pyplot as plt


def plot_policy_heatmap(model, env, n_bins=50, title="Policy Heatmap", save_path=None):
    """
    Visualize the greedy policy over the discretized state space.
    Works for discrete-action models (DQN, Q-table, etc.)
    """
    pos_space = np.linspace(-1.2, 0.6, n_bins)
    vel_space = np.linspace(-0.07, 0.07, n_bins)

    policy_grid = np.zeros((n_bins, n_bins))
    for i, pos in enumerate(pos_space):
        for j, vel in enumerate(vel_space):
            obs = np.array([[pos, vel]])
            action, _ = model.predict(obs, deterministic=True)
            policy_grid[j, i] = action

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        policy_grid,
        origin="lower",
        extent=[-1.2, 0.6, -0.07, 0.07],
        aspect="auto",
        cmap="coolwarm",
    )
    plt.colorbar(im, ax=ax, label="Action (0=left, 1=idle, 2=right)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_policy(model, env, n_episodes=100, seed=42):
    """
    Run n_episodes and return a dict of summary stats.
    """
    rewards, lengths, successes = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total_reward, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(int(terminated))  # terminated = reached goal

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(lengths),
        "std_steps": np.std(lengths),
        "success_rate": np.mean(successes),
    }


def plot_training_curve(log_path, title="Training Curve", save_path=None):
    """
    Plot episode reward over training steps from a Monitor log CSV.
    """
    import pandas as pd
    df = pd.read_csv(log_path, skiprows=1)
    df["cumulative_steps"] = df["l"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["cumulative_steps"], df["r"], alpha=0.4, label="Episode reward")
    ax.plot(
        df["cumulative_steps"],
        df["r"].rolling(50).mean(),
        color="red",
        label="50-ep moving avg",
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
