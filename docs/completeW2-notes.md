# Week 2 — Complete Notes

# MARL Foundations & RL-Based Coordination

---

# Day 1 — Introduction to MARL

## What is MARL?

MARL (Multi-Agent Reinforcement Learning) studies how multiple agents learn and interact inside shared environments.

Unlike single-agent RL:

* agents affect each other’s observations,
* rewards,
* and environment dynamics.

---

# RL vs MARL

| RL                            | MARL                      |
| ----------------------------- | ------------------------- |
| Single agent                  | Multiple agents           |
| Mostly stationary environment | Dynamic environment       |
| Simpler learning              | More complex coordination |

---

# Types of MARL Systems

## Cooperative MARL

Agents work together for shared rewards.

Example:

* robot teams,
* drone swarms.

---

## Competitive MARL

Agents compete against each other.

Example:

* chess AI,
* game-playing agents.

---

# Main Intuition

In MARL:

```text id="w2n01"
other agents become part of the environment itself.
```

This makes learning harder because:

* behaviors constantly change,
* environment dynamics become non-stationary.

---

# Day 2 — PettingZoo Environments

## What is PettingZoo?

PettingZoo is a Python library for:

# multi-agent reinforcement learning environments.

Similar to Gym, but designed for:

* multiple agents,
* sequential interactions,
* MARL workflows.

---

# Main Components

## Observation

Information received by an agent.

---

## Action Space

Valid actions an agent can perform.

---

## Reward

Feedback given by environment.

---

## Environment Step

```text id="w2n02"
Observation → Action → Reward → Next State
```

---

# Action Masks

PettingZoo often provides:

# valid action masks.

Example:

```python id="w2n03"
[1,1,0,1]
```

`0` means:

```text id="w2n04"
invalid action
```

---

# Main Intuition

PettingZoo provides:

# realistic MARL interaction loops.

---

# Day 3 — Sender–Receiver Communication

## Sender–Receiver System

One agent:

# observes information.

Another agent:

# depends on communication.

---

# Example

Sender sees:

```text id="w2n05"
target location
```

Receiver cannot see target directly.

Sender communicates:

* symbol,
* message,
* signal.

Receiver acts using:

# received information.

---

# Main Intuition

```text id="w2n06"
Communication becomes valuable when agents have incomplete information individually.
```

---

# Communication Protocol

A protocol is:

# the rule system agents use for communication.

Example:

| Symbol | Meaning |
| ------ | ------- |
| 0      | LEFT    |
| 1      | RIGHT   |

---

# Day 4 — Communication Stabilization

## Stable Communication

Consistent communication mappings.

Example:

| Goal  | Message |
| ----- | ------- |
| LEFT  | 0       |
| RIGHT | 1       |

---

# Unstable Communication

Random or changing communication.

Example:

| Goal  | Message |
| ----- | ------- |
| LEFT  | random  |
| RIGHT | random  |

---

# Reward Reinforcement

Successful communication:

* survives,
* stabilizes,
* gets reused.

Failed communication:

* changes,
* disappears.

---

# Main Intuition

```text id="w2n07"
Useful communication patterns stabilize because they consistently improve rewards.
```

---

# Day 5 — Noisy Communication & Limited Vocabulary

# Noisy Communication

Messages may become corrupted during transfer.

Example:

```text id="w2n08"
Sender sends 0
Receiver receives 1
```

This causes:

* coordination failures,
* incorrect actions.

---

# Limited Vocabulary Systems

Agents communicate using:

# very few symbols.

Example:

```text id="w2n09"
{0,1}
```

---

# Main Intuition

Limited communication forces:

# efficient symbolic communication.

---

# Why This Matters

Real systems often have:

* bandwidth limits,
* memory limits,
* communication constraints.

---

# Day 6 — Communication Bottlenecks

## What is a Communication Bottleneck?

When:

# information to communicate > communication capacity.

Example:

Sender must communicate:

* LEFT
* RIGHT
* UP
* DOWN

using only:

```text id="w2n10"
0 or 1
```

---

# Information Compression

Agents must:

# compress information efficiently.

---

# Ambiguous Communication

Multiple states may share:

# same symbol.

Example:

| Goal | Message |
| ---- | ------- |
| LEFT | 0       |
| UP   | 0       |

Receiver cannot perfectly identify:

# exact state.

---

# Main Intuition

```text id="w2n11"
Communication bottlenecks create information loss and ambiguity.
```

---

# Day 7 — RL-Based Learning Agents

# Q-Learning

Q-learning learns:

Q(s,a)

Meaning:

# value of taking action in a state.

---

# Q-Learning Update

Core learning equation:

Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma\max Q(s',a')-Q(s,a)]

---

# Important Terms

## Alpha (α)

Learning rate.

Controls:

# how strongly new information updates Q-values.

---

## Gamma (γ)

Discount factor.

Controls:

# importance of future rewards.

---

# Exploration vs Exploitation

## Exploration

Trying random actions.

---

## Exploitation

Using best known action.

---

# Learned vs Random Agents

Learned agents:

* optimize rewards,
* improve behavior over time.

Random agents:

* behave inconsistently,
* lack optimization.

---

# Main Intuition

```text id="w2n12"
Agents improve behavior by updating action values using repeated reward feedback.
```

---

# Overall Week 2 Understanding

By the end of Week 2:

* MARL foundations were established,
* communication systems were introduced,
* RL learning intuition became clear,
* environment-based coordination was understood,
* and the transition toward emergent communication systems began.

This week formed the foundation for:

* trainable communication systems,
* cooperative MARL environments,
* and emergent communication research.
