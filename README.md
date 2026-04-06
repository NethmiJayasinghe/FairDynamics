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
### Key Findings
- CQL reduces EO gap by **~84%**
- CQL reduces DP gap by **~79%**
- Achieves **near-equalized TPR across groups**
- Matches performance of a strong greedy baseline
- Outperforms all non-adaptive methods

---
### How to Run

This project is implemented as a Jupyter notebook.

1. Install basic dependencies:
Make sure you have Python and Jupyter installed.
2. Launch Jupyter Notebook
3. Run all cells
