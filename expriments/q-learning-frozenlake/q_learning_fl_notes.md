# Q-Learning in FrozenLake — Notes

## Objective
Understand:
- how RL agents start learning,
- Q-values,
- Bellman-based updates,
- exploration vs exploitation,
- and gradual policy improvement.

---

# Q-Learning

Q-Learning is a value-based Reinforcement Learning algorithm where the agent learns:

Q(s, a)

Meaning:
“How good is taking action a in state s?”

The agent gradually improves these values through repeated interaction with the environment.

---

# Q-Table

Q-values are stored in a table called:
Q-Table.

Rows:
- states

Columns:
- actions

Each value represents:
the estimated long-term usefulness of taking an action in a state.

---

# FrozenLake Environment

Environment:
FrozenLake-v1

Used with:
is_slippery=False

to make transitions deterministic and easier for observing learning behavior.

---

# Important Hyperparameters

## Alpha (α) — Learning Rate
Controls:
how strongly new experiences update Q-values.

Small alpha:
slow learning.

Large alpha:
fast but unstable learning.

---

## Gamma (γ) — Discount Factor
Controls:
importance of future rewards.

Small gamma:
short-term focus.

Large gamma:
long-term reward focus.

---

## Epsilon (ε)
Controls exploration probability.

Agent:
- explores randomly sometimes,
- exploits best-known actions otherwise.

---

# Q-Learning Update Equation

Q(s,a) = Q(s,a) + α[Reward + γ max Q(s',a') - Q(s,a)]

Intuition:
New estimate =
old estimate +
correction using:
- reward
- future best possible value

---

# Bellman Intuition

Current actions become valuable because:
they may lead to better future states and future rewards.

The agent learns long-term behavior instead of immediate reward optimization.

---

# Important Learning Process

Initially:
- Q-values are zero,
- agent behaves randomly.

Over repeated episodes:
- rewards update Q-values,
- better actions receive higher values,
- policy gradually improves.

Learning emerges through repeated value updates.

---

# Key Learnings

- RL agents improve through experience.
- Q-values estimate action usefulness.
- Bellman recursion drives learning.
- Exploration is necessary to discover better strategies.
- Future rewards influence current decisions.
- Policies improve gradually over time.

---

# Important Understanding

This experiment demonstrates the transition from:
random behavior
→
learning-based intelligent behavior using value updates and future reward estimation.
