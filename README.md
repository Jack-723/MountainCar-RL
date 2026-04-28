# RLI Group Assignment — MountainCar

**Course:** Reinforcement Learning  
**Assignment:** Group Project 

## Team
Jack · Omar · Salmane · Matthew · Enrique · Leena · Nathan

---

## Overview

This project analyzes and compares optimal RL policies for four variations of the MountainCar Gymnasium environment. The goal is to identify the best RL solution for each problem variant, comparing algorithms, state representations, reward structures, and resulting policies.

---

## Repository Structure

```
mountaincar-rl/
│
├── notebooks/
│   ├── scenario_1_discrete_min_steps.ipynb      # MountainCar-v0 — minimize steps
│   ├── scenario_2_continuous_default.ipynb      # MountainCarContinuous-v0 — default reward
│   ├── scenario_3_discrete_min_fuel.ipynb       # MountainCar-v0 (adapted) — minimize fuel
│   └── scenario_4_continuous_min_steps.ipynb    # MountainCarContinuous-v0 (adapted) — minimize non-null actions
│
├── src/
│   ├── envs/           # Custom Gym wrappers for adapted reward scenarios
│   ├── agents/         # Agent training/evaluation logic
│   └── utils/          # Plotting, logging, evaluation helpers
│
├── results/
│   ├── scenario_1/     # Saved models, plots, metrics
│   ├── scenario_2/
│   ├── scenario_3/
│   └── scenario_4/
│
├── presentation/       # Final PDF/PPTX submission
├── refs/               # Papers and references for Part 02
├── requirements.txt
└── README.md
```

---

## Scenarios

| # | Environment | Reward / Objective |
|---|---|---|
| 1 | `MountainCar-v0` | Default — minimize steps to goal |
| 2 | `MountainCarContinuous-v0` | Default — penalize large action magnitudes |
| 3 | `MountainCar-v0` (adapted) | Minimize fuel — cost ∝ number of non-null actions |
| 4 | `MountainCarContinuous-v0` (adapted) | Minimize "real time" — cost ∝ number of non-null force applications (`-1` per step where `|a| > 0.01`, plus `+100` goal bonus) |

---

## Setup

```bash
git clone https://github.com/<your-username>/mountaincar-rl.git
cd mountaincar-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

---

## Submission

One ZIP submitted by the group coordinator, named:  
`RLI_22_00 – Group {XY}.zip`
