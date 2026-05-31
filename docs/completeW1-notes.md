# Week 1 — Reinforcement Learning Foundations

---

# Day 1 — Introduction to Reinforcement Learning

## Topics Covered

* What is Reinforcement Learning?
* Agents and environments
* States, actions, and rewards
* Episodes and trajectories
* Exploration vs exploitation
* FrozenLake environment basics
* Deterministic vs stochastic environments

---

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of Machine Learning where an agent learns by interacting with an environment.

The agent:

* performs actions,
* receives rewards or penalties,
* and gradually learns better behavior.

The goal of the agent is:

* maximize long-term cumulative reward.

---

## Core RL Components

### Agent

The decision maker.

Example:

* robot,
* game player,
* self-driving car.

---

### Environment

The world in which the agent operates.

Example:

* FrozenLake,
* Taxi-v3,
* Chess,
* Atari games.

---

### State

Current situation of the environment.

Example:

* current cell position,
* board configuration,
* robot coordinates.

---

### Action

A move or decision taken by the agent.

Examples:

* left,
* right,
* pickup,
* dropoff.

---

### Reward

Feedback signal.

Positive reward:

* good action.

Negative reward:

* bad action.

---

### Episode

One complete interaction sequence from:

* start state
* to terminal state.

---

## Exploration vs Exploitation

### Exploration

Trying new actions.

Purpose:

* discover better paths.

---

### Exploitation

Using already learned good actions.

Purpose:

* maximize reward.

---

## FrozenLake Environment

Grid-world environment where:

* agent starts at beginning,
* goal must be reached,
* holes must be avoided.

---

## Deterministic vs Stochastic

### Deterministic

Same action always gives same result.

Example:

```python
is_slippery=False
```

---

### Stochastic

Same action may produce different outcomes.

Example:

```python
is_slippery=True
```

---

# Day 2 — Markov Decision Process and Bellman Equation

## Topics Covered

* Markov Decision Process (MDP)
* State transitions
* Discount factor
* Value functions
* Bellman Equation intuition
* Future rewards

---

## Markov Decision Process (MDP)

An RL problem is mathematically modeled as an MDP.

An MDP contains:

* States
* Actions
* Rewards
* Transition probabilities
* Discount factor

---

## Markov Property

Future depends only on:

* current state,
  not full history.

---

## Discount Factor

Symbol:

```text
gamma
```

Controls importance of future rewards.

### High gamma

Future rewards important.

### Low gamma

Immediate rewards important.

---

## Value Function

Represents:

* how good a state is.

Symbol:

```text
V(s)
```

---

## Bellman Equation

Core RL idea:

```text
Current Value = Reward + Discounted Future Value
```

Bellman intuition:

* good states lead to future good states.

---

## Important Understanding

RL agents learn by:

* estimating future rewards,
* improving state values,
* and choosing better actions over time.

---

# Day 3 — Q-Learning

## Topics Covered

* Q-table
* Q-values
* Q-learning update rule
* Epsilon-greedy policy
* Policy improvement
* Taxi-v3 environment

---

## Q-Table

Stores:

```text
Q(state, action)
```

Each value represents:

* long-term usefulness of taking an action from a state.

---

## Q-Learning

Q-learning is an off-policy Temporal Difference learning algorithm.

The agent learns:

* optimal action values.

---

## Q-Learning Update Equation

```text
Q(s,a)=Q(s,a)+alpha[reward+gamma maxQ(s',a')-Q(s,a)]
```

---

## Important Understanding

Q-learning updates current values using:

* reward,
* estimated best future action value.

---

## Epsilon-Greedy Exploration

### Exploitation

Choose highest Q-value action.

### Exploration

Choose random action.

Controlled using:

```text
epsilon
```

---

## Taxi-v3 Environment

Task:

* pickup passenger,
* navigate grid,
* drop passenger correctly.

Sparse rewards are used.

The agent initially has:

* no intelligence,
* no map knowledge,
* no strategy.

Learning gradually emerges from:

* exploration,
* rewards,
* repeated interaction.

---

# Day 4 — Temporal Difference (TD) Learning

## Topics Covered

* Temporal Difference learning
* TD error
* Online learning
* Difference from Monte Carlo
* Policy evaluation intuition

---

## What is TD Learning?

TD Learning updates values:

* during interaction,
* before episode ends.

---

## Core TD Idea

Learn immediately using:

* current reward,
* estimated future value.

---

## TD Error

```text
TD Error = Reward + gamma(Future Estimate) - Current Estimate
```

Represents:

* correction signal.

---

## Main Advantage

Unlike Monte Carlo:

* TD does not wait for episode completion.

This enables:

* faster learning,
* online updates,
* continuous learning.

---

## Important Understanding

SARSA and Q-learning are both:

* TD learning algorithms.

---

# Day 5 — SARSA and Monte Carlo Learning

## Topics Covered

* SARSA
* On-policy learning
* Monte Carlo learning
* Return calculation
* On-policy vs Off-policy

---

# SARSA

SARSA stands for:

```text
State → Action → Reward → State → Action
```

---

## SARSA Update Equation

```text
Q(s,a)=Q(s,a)+alpha[reward+gammaQ(s',a')-Q(s,a)]
```

---

## Important Difference from Q-Learning

### Q-learning

Uses:

* best future action value.

---

### SARSA

Uses:

* actual next selected action.

---

## On-Policy Learning

SARSA is on-policy because:

* acting policy
  and
* learning policy
  are the same.

---

## Q-Learning vs SARSA

### Q-Learning

* aggressive
* optimal future assumption
* off-policy

### SARSA

* conservative
* exploration-aware
* on-policy

---

# Monte Carlo Learning

Monte Carlo learning:

* waits until episode ends,
* then updates values.

---

## Monte Carlo Return

```text
G_t = R_(t+1)+gammaR_(t+2)+gamma²R_(t+3)+...
```

---

## Important Understanding

Monte Carlo uses:

* actual final outcomes.

No future estimation is used during interaction.

---

## Monte Carlo vs TD

### Monte Carlo

* episode-based
* actual returns
* slower learning

### TD Learning

* online learning
* estimated future values
* faster updates

---

# Day 6 — Deep Reinforcement Learning Intuition

## Topics Covered

* Q-table limitations
* Curse of dimensionality
* Function approximation
* Neural network intuition
* Deep Q Networks (DQN)
* Experience replay
* Target networks

---

## Why Q-Tables Fail

Q-tables work only for:

* small environments.

Large environments create:

* memory explosion,
* huge state spaces,
* impossible storage requirements.

---

## Curse of Dimensionality

As states increase:

* Q-table size grows exponentially.

Real-world environments become impossible to memorize.

---

## Main Limitation

Q-tables:

* memorize exact states.

They do NOT:

* generalize patterns.

---

## Function Approximation

Deep RL replaces:

* exact table storage

with:

* neural network approximation.

---

## Neural Network Intuition

Input:

* environment state.

Output:

* predicted Q-values.

Neural networks learn:

* patterns,
* relationships,
* generalized behavior.

---

## Deep Q Networks (DQN)

Main idea:

```text
Replace Q-table with neural network.
```

The Bellman/Q-learning logic still remains conceptually similar.

---

## Experience Replay

Stores experiences:

```text
(state, action, reward, next_state)
```

Allows:

* stable training,
* reduced correlation,
* better learning.

---

## Target Networks

Used to:

* stabilize learning,
* avoid rapidly changing targets.

---

## Important Understanding

### Classical RL

* explicit memorization
* table lookup

### Deep RL

* generalized prediction
* neural network approximation

---

# Day 7 — RL Workflow Consolidation and PyTorch Basics

## Topics Covered

* RL workflow revision
* Policy improvement intuition
* Convergence intuition
* Q-learning vs SARSA comparison
* PyTorch basics
* Tensor basics

---

## RL Workflow

```text
Environment → State → Action → Reward → Next State → Update → Policy Improvement
```

This forms the complete Reinforcement Learning cycle.

---

## Policy Improvement

Policies improve gradually because:

* Q-values change,
* action preferences improve,
* rewards guide learning.

---

## Convergence

Over many episodes:

* values stabilize,
* policies improve,
* exploration reduces.

---

## PyTorch Introduction

PyTorch is a Deep Learning framework used for:

* neural networks,
* tensors,
* Deep RL systems.

---

## What is a Tensor?

Tensor = multidimensional numerical array.

Examples:

* scalars,
* vectors,
* matrices,
* higher-dimensional structures.

---

## Tensor Operations

Basic operations:

* addition,
* multiplication,
* matrix multiplication,
* dot products.

---

## Why Tensors Matter in Deep RL

Later in Deep RL:

* states,
* rewards,
* neural network inputs,
* Q-values,
* observations

all become tensors.

---

## Neural Network Intuition

Input state → neural network → predicted Q-values.

Instead of storing values directly:

* weights learn patterns,
* Q-values are predicted dynamically.

---

# Final Week 1 Understanding

By the end of Week 1:

* Reinforcement Learning fundamentals were understood.
* Q-learning and SARSA were implemented.
* Monte Carlo and TD learning were explored.
* On-policy vs Off-policy learning was understood.
* Deep RL intuition and Q-table limitations were studied.
* PyTorch and tensor foundations for Deep RL were introduced.

This forms the foundational base required before moving into:

* Multi-Agent Reinforcement Learning (MARL)
* Emergent Communication
* Deep RL architectures
* Sender-Receiver systems
* Multi-agent communication experiments.
