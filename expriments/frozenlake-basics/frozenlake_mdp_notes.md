# FrozenLake-v1 — MDP Intuition Notes

## Objective
Understand:
- Markov Decision Processes (MDPs)
- state transitions
- stochastic/probabilistic environments
- delayed rewards
- long-term decision making

---

# Environment Overview

FrozenLake is a grid-world environment where:
- the agent starts at S (start),
- must reach G (goal),
- while avoiding H (holes).

The environment is slippery, meaning actions may not always behave exactly as intended.

---

# Environment Symbols

S → Start  
F → Frozen safe tile  
H → Hole (failure state)  
G → Goal state  

---

# Actions

0 → Left  
1 → Down  
2 → Right  
3 → Up  

---

# MDP Components

## State
The current grid position of the agent.

## Action
Movement chosen by the agent.

## Transition
Current state + action → next state.

Because the environment is slippery:
the same action may lead to different next states.

Example:
Choosing RIGHT may:
- move right,
- slip upward,
- slip downward.

This demonstrates stochastic/probabilistic transitions.

---

# Transition Probability

Theoretical notation:

P(s' | s, a)

Meaning:
Probability of reaching next state s'
given:
- current state s
- action a

---

# Reward Structure

Normal movement → 0  
Falling into hole → 0  
Reaching goal → 1  

This is a sparse reward environment because rewards occur very rarely.

---

# Important RL Intuitions

- Actions affect future states.
- Same action may produce different outcomes.
- RL environments can be stochastic.
- Immediate rewards may not exist.
- Long-term planning is important.
- Random agents perform poorly.
- Learning is necessary to consistently reach the goal.

---

# Markov Property

The current state contains enough information for future decision making.

Past movement history is unnecessary if the current state fully describes the environment.

---

# Why Total Reward Often Remains 0

Random actions usually:
- wander aimlessly,
- fall into holes,
- fail to reach the goal.

Since reward is only given at the goal,
random agents often obtain:
Total Reward = 0

This demonstrates the need for Reinforcement Learning algorithms.

---

# Key Learning

FrozenLake demonstrates:
- sequential decision making,
- stochastic state transitions,
- delayed rewards,
- uncertainty in environments,
- and why RL agents must learn optimal behavior instead of acting randomly.
