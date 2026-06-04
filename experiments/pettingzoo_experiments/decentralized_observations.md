# Decentralized Observation Experiment

## Objective

The objective of this experiment is to understand decentralized observations and local information processing inside Multi-Agent Reinforcement Learning environments.

This experiment demonstrates how different agents receive:

* their own observations,
* local environment information,
* and valid action restrictions.

---

# What the Experiment Does

The experiment:

* runs a PettingZoo TicTacToe environment,
* cycles through agents,
* prints observation shapes,
* displays action masks,
* and allows random actions.

Each agent receives:

* a local observation tensor,
* and an action mask defining valid moves.

---

# Main Concepts Demonstrated

## Decentralized Observations

Each agent receives its own observation instead of a shared global observation.

---

## Local Information Processing

Agents interact using encoded observations rather than raw environment access.

---

## Action Masks

Action masks restrict invalid actions and maintain valid environment behavior.

---

# Observations

* Different agents receive decentralized observations.
* Observation spaces are agent-specific.
* Action masks define legal and illegal actions.
* Agents interact using local information only.
* Environment constraints affect available actions.

---

# Important Understanding

Agents in MARL environments often:

* do not see the full environment directly,
* and instead operate using local observations.

This creates:

* decentralized decision making,
* incomplete information,
* and coordination challenges.

---

# Final Understanding

This experiment demonstrates how real MARL systems operate using:

* local observations,
* decentralized interaction,
* and constrained action spaces.
