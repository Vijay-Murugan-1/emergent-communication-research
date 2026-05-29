SARSA on FrozenLake

Objective

Implement and understand SARSA (State-Action-Reward-State-Action) using the FrozenLake-v1 environment.

---

Key Concepts

- On-policy Reinforcement Learning
- Temporal Difference (TD) Learning
- Q-table updates
- Epsilon-greedy exploration
- Exploration-aware learning

---

SARSA Update Equation

Q(s,a) = Q(s,a) + α [ reward + γQ(s',a') - Q(s,a) ]

Where:

- "s" = current state
- "a" = current action
- "s'" = next state
- "a'" = actual next action selected by policy
- "α" = learning rate
- "γ" = discount factor

---

Core Idea

SARSA updates Q-values using the actual next action chosen by the current policy.

Unlike Q-learning, SARSA does not assume optimal future behavior during updates.

This makes SARSA:

- exploration-aware
- safer in risky environments
- more conservative than Q-learning

---

On-Policy Learning

SARSA is called an on-policy algorithm because:

- the acting policy
  and
- the learning policy

are the same.

The agent learns from the exact behavior policy currently being followed.

---

Environment

Environment used:

- FrozenLake-v1
- "is_slippery=False"

The agent learns to move from start state to goal state while avoiding holes.

---

Observations

- Q-values improve gradually with episodes.
- Exploration directly affects updates.
- Different runs may produce slightly different Q-tables.
- Hyperparameters strongly affect convergence and stability.

---

Important Difference from Q-Learning

Q-learning update:

- uses maximum future Q-value
- assumes future optimal behavior

SARSA update:

- uses actual next selected action
- learns under real exploratory behavior
