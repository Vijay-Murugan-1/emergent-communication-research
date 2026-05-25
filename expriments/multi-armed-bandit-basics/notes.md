# Multi-Armed Bandit — Notes

## Core Idea
Agent chooses among multiple actions without knowing which is best initially and learns through rewards.

---

## Exploration
Trying random/new actions to gather information.

## Exploitation
Using the currently best-known action.

## Exploration vs Exploitation
Balance between:
- discovering better actions
- maximizing current reward

---

## Action-Value Estimate

Q(a):
Expected reward of action `a`.

Agent gradually updates estimates from experience.

---

## Reward Assignment

Each action has a hidden reward probability.

Example:
- Bandit 0 → 20%
- Bandit 1 → 50%
- Bandit 2 → 75%
- Bandit 3 → 30%

Rewards are probabilistic/stochastic.

---

## Incremental Update Rule

Q_new = Q_old + learning_rate * (Reward - Q_old)

Meaning:
- old estimate
- corrected using new experience

(Reward - Q_old)
→ prediction error

---

## Epsilon-Greedy

epsilon:
probability of random exploration.

Example:
epsilon = 0.1
- 90% exploit
- 10% explore

---

## Key Learnings

- Agent initially knows nothing.
- Learning happens through interaction and rewards.
- Estimates improve gradually over time.
- Exploration is necessary to discover better actions.
- RL learning is based on iterative prediction correction.
