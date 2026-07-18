# Taxi-v3 — MDP Intuition Notes

## Objective
Understand:
- Markov Decision Processes (MDPs)
- sequential decision making
- state transitions
- delayed rewards
- long-term reward optimization
- Bellman intuition

---

# Environment Overview

Taxi-v3 is an RL environment where:
- a taxi must pick up a passenger,
- navigate the environment,
- and successfully drop the passenger at the destination.

The environment demonstrates long-term planning and sequential decision making.

---

# Actions

0 → Move South  
1 → Move North  
2 → Move East  
3 → Move West  
4 → Pickup Passenger  
5 → Drop Passenger  

---

# Reward Structure

Normal movement → -1  
Wrong pickup/drop → -10  
Successful dropoff → +20  

---

# Important Reward Intuition

Movement gives negative reward:
- to encourage efficiency,
- prevent random wandering.

Even though movement gives:
- immediate negative reward,
it may still improve future states and future rewards.

This demonstrates long-term optimization.

---

# MDP Components

## State
The current environment condition.

State internally contains:
- taxi position,
- passenger location,
- destination.

The environment provides the initial state using:

env.reset()

---

## Action
Decision taken by the agent.

Actions change future states.

---

## Transition

Current State + Action → Next State

The environment computes:
- next state,
- reward,
- termination condition

after every action.

---

## Reward
Feedback from environment.

Rewards define:
- desired behavior,
- optimization objective.

---

## Policy
The strategy used by the agent.

Policy answers:
“In this state, what action should I take?”

---

# RL Interaction Loop

State →
Action →
Environment Transition →
Reward →
Next State

This loop repeats continuously.

---

# Important RL Intuitions

- RL is sequential decision making.
- Actions affect future possibilities.
- Immediate rewards may be negative.
- Long-term future rewards matter more.
- Current actions influence future states.
- Agents optimize cumulative future reward.

---

# Bellman Intuition

Current state/action value depends on:
- immediate reward
PLUS
- future rewards that follow.

Example:
Moving toward passenger:
Immediate reward = -1

But:
future possibility improves,
eventually leading to +20 reward.

Thus movement actions can still have high long-term value.

---

# Markov Property

Current state contains enough information for decision making.

Past movement history is unnecessary if the current state fully describes the environment.

---

# Why Taxi-v3 is Important

Taxi-v3 clearly demonstrates:
- long-term planning,
- delayed rewards,
- sequential state transitions,
- Bellman intuition,
- and why RL agents optimize future cumulative rewards instead of immediate rewards alone.

---

# Key Learning

Taxi-v3 demonstrates that:
good actions are not always immediately rewarding.

Their true value comes from:
- future consequences,
- future states,
- and future rewards.
