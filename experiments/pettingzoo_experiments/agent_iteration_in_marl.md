# PettingZoo Environment Exploration

## Objective

The objective of this experiment is to understand the basic workflow of a real Multi-Agent Reinforcement Learning environment using PettingZoo.

This experiment introduces:

* agent iteration,
* environment stepping,
* multi-agent interaction loops,
* and sequential MARL execution.

---

# What the Experiment Does

The environment used is:

* TicTacToe from PettingZoo.

The experiment:

* creates the environment,
* resets the game,
* cycles through agents,
* provides observations and rewards,
* and allows agents to take random actions.

The environment automatically alternates turns between agents until:

* the game ends,
* or termination occurs.

---

# Main Concepts Demonstrated

## Agent Iteration

The environment continuously cycles through multiple agents one after another.

---

## Sequential Interaction

Agents act sequentially instead of simultaneously.

---

## Multi-Agent Workflow

Each agent:

* receives observations,
* takes actions,
* receives rewards,
* and affects environment state.

---

# Observations

* Multiple agents interact inside the same environment.
* Agents receive turns sequentially.
* Environment dynamics change after every action.
* Rewards belong to individual agents.
* Environment controls interaction order automatically.

---

# Important Understanding

This experiment demonstrates how real Multi-Agent Reinforcement Learning environments differ from single-agent RL systems.

Instead of:

* one observation stream,
* one action loop,

there are now:

* multiple agents,
* multiple observations,
* and shared environment dynamics.

---

# Final Understanding

This experiment forms the foundational understanding of:

* real MARL environment interaction,
* agent iteration systems,
* and decentralized multi-agent workflows.
