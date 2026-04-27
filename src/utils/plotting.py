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


def plot_phase_portrait(model, env, n_trajectories=5, max_steps=200, seed=42,
                        title="Phase Portrait", save_path=None):
    """Overlay successful trajectories on the (position, velocity) plane."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Background: position vs. flag line.
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Goal (pos=0.5)")
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.5)

    cmap = plt.cm.viridis
    drawn = 0
    attempt_seed = seed
    while drawn < n_trajectories and attempt_seed < seed + 50:
        obs, _ = env.reset(seed=attempt_seed)
        attempt_seed += 1
        positions, velocities = [obs[0]], [obs[1]]
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            positions.append(obs[0])
            velocities.append(obs[1])
            steps += 1
        if terminated:
            color = cmap(drawn / max(1, n_trajectories - 1))
            ax.plot(positions, velocities, color=color, alpha=0.85,
                    linewidth=1.2, label=f"Run {drawn + 1} ({steps} steps)")
            ax.scatter(positions[0], velocities[0], color=color, marker="o",
                       s=30, zorder=5)
            drawn += 1

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.07, 0.07)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return drawn


def plot_training_curves_multi(monitor_paths, title="Training Curves",
                               save_path=None, rolling=50):
    """Overlay episode-reward curves from multiple SB3 Monitor CSVs."""
    import pandas as pd

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, path in monitor_paths.items():
        if not path or not __import__("os").path.exists(path):
            continue
        df = pd.read_csv(path, skiprows=1)
        df["cum"] = df["l"].cumsum()
        ax.plot(df["cum"], df["r"].rolling(rolling, min_periods=1).mean(),
                label=label, linewidth=1.5)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Episode reward (rolling-{rolling} mean)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_metrics_table_png(df, save_path, title="Scenario 1 — Evaluation Metrics"):
    """Render a pandas DataFrame as a PNG suitable for slides."""
    fig, ax = plt.subplots(figsize=(min(2 + 1.4 * len(df.columns), 14),
                                    0.55 + 0.4 * len(df)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    table = ax.table(cellText=df.round(3).values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


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
