# Adaptive Fairness-Constrained Decision Making via Offline Reinforcement Learning

This project explores how **offline reinforcement learning (RL)** can be used to learn **adaptive fairness interventions** in non-stationary environments. We formulate fairness mitigation as a sequential decision-making problem and train a **Conservative Q-Learning (CQL)** agent to dynamically adjust decision thresholds across demographic groups.

---

## Motivation

Most fairness interventions in machine learning (e.g., threshold tuning, reweighting) are **static**. However, real-world systems are often **non-stationary**, where demographic distributions and data characteristics evolve over time.

As a result:
- Static fairness corrections degrade over time  
- Fairness–accuracy trade-offs shift dynamically  
- One-time interventions are insufficient  

This project addresses this by learning **adaptive policies** using offline RL.

---

## Key Idea

We model fairness intervention as a **Markov Decision Process (MDP)**:

- **State:**  
  Classifier performance + subgroup fairness metrics  
  (accuracy, TPR per group, EO gap, DP gap, demographic proportions, time)

- **Action:**  
  Discrete threshold adjustments for each demographic group

- **Reward:**  
  Balances accuracy and fairness R = (1 − λ) · Accuracy − λ · EO_gap

- **Environment:**  
Simulated fairness environment with **demographic drift**

---
## Visualizations

The project includes:
- Agent comparison plots (reward, fairness, accuracy)
- Fairness–accuracy trade-off curves
- Time-series analysis under demographic drift
- Per-group TPR comparisons
- Policy trajectory visualization

---
## Approach

- **MDP** — State: 10-dim (accuracy, per-group TPR, EO/DP gaps, group proportions, time). Action: discrete threshold adjustments per group. Drift: sinusoidal demographic shift every 10 steps.
- **Reward**:
  ```
  R = (1−λ)·Accuracy − λ·FairnessGap
      + 0.15·min_g TPR_g          (min-TPR floor)
      − 0.05·Σ_g max(0, 0.35 − t_g)  (threshold-bound penalty)
  ```
  *FairnessGap = EO gap on COMPAS, DP gap on Adult — chosen from the AIF360 baseline audit.*
- **Agent** — CQL with double Q-learning, prioritized replay, soft Polyak target updates (τ=0.005), 100k gradient steps, α=0.5.
- **Behavior policies (offline data)** — 5 scripted policies (random, no-intervention, sweep, balanced, greedy) generate 60,000 transitions per dataset.

---

## Key Results

| Dataset | Agent | Reward | Accuracy | Fairness Gap |
|---|---|---|---|---|
| COMPAS | **CQL (ours)** | **19.11** | 0.551 | **EO 0.033** |
| COMPAS | Greedy | 18.85 | 0.550 | EO 0.039 |
| COMPAS | No Intervention | 12.95 | **0.657** | EO 0.275 |
| Adult Income | **CQL (ours)** | **14.34** | 0.752 | **DP 0.026** |
| Adult Income | Greedy | 14.14 | 0.730 | DP 0.033 |
| Adult Income | No Intervention | 11.77 | **0.802** | DP 0.083 |

- **COMPAS:** EO gap reduced ~88% vs no intervention; CQL beats Greedy on both reward and EO gap
- **Adult Income:** DP gap reduced ~69% vs no intervention; CQL beats Greedy on reward, accuracy, AND DP gap simultaneously
- The same CQL pipeline transfers across datasets — only the fairness target metric and λ differ

---
### How to Run

This project is implemented as a Jupyter notebook.

1. Install basic dependencies:
Make sure you have Python and Jupyter installed.
2. Launch Jupyter Notebook
3. Run all cells
