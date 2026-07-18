# Q-Learning vs SARSA Comparison

## Objective
Compare the behavior and learning differences between Q-learning and SARSA using the FrozenLake-v1 environment.

---

## Environment
- FrozenLake-v1
- Deterministic environment
- is_slippery=False

---

## Q-Learning
Q-learning is an off-policy Temporal Difference learning algorithm that updates Q-values using the maximum possible future Q-value.

Core idea:
Learn assuming optimal future behavior.

---

## SARSA
SARSA is an on-policy Temporal Difference learning algorithm that updates Q-values using the actual next action selected by the current policy.

Core idea:
Learn using actual exploratory behavior.

---

## Main Difference

Q-Learning:
- aggressive learning
- optimal future assumption
- off-policy

SARSA:
- exploration-aware learning
- conservative behavior
- on-policy

---

## Observations
- Both algorithms improve policies over repeated episodes.
- Q-learning may converge faster.
- SARSA often learns safer and more conservative policies.
- Exploration directly affects SARSA updates.
- Different runs may produce slightly different Q-tables due to stochastic exploration.

---

## Important Understanding

Q-Learning asks:
“What is the best possible future action?”

SARSA asks:
“What action did I actually choose next?”

This creates the core difference between off-policy and on-policy learning.
