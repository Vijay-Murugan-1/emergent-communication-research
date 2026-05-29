This experiment implements Monte Carlo State Value Estimation on the FrozenLake-v1 environment using Reinforcement Learning.

Monte Carlo learning is one of the foundational Reinforcement Learning methods where learning occurs only after the complete episode finishes. Instead of estimating future values during interaction, Monte Carlo learning uses actual observed returns from completed episodes.

The main objective of this experiment is to understand:
- Episode-based learning
- Return calculation
- Discounted rewards
- State value estimation
- Learning from complete outcomes

Environment:
FrozenLake-v1 is a grid-world environment where the agent navigates from the start state to the goal while avoiding holes.

In this implementation:
is_slippery=False

which creates deterministic state transitions.

State Values:
The algorithm stores value estimates for states:
V(s)

Each value represents the expected long-term reward obtainable from that state.

Monte Carlo Return:
Monte Carlo learning calculates the total discounted future reward after the episode completely ends.

Return Formula:

G_t = R_(t+1) + gamma*R_(t+2) + gamma²*R_(t+3) + ...

Where:
- G_t = return from current state
- gamma = discount factor
- R = future rewards

The algorithm processes the episode backward and repeatedly applies:

G = reward + gamma * G

This recursive update automatically creates exponential discounting of future rewards.

Monte Carlo Update Rule:

V(s) = V(s) + alpha * (G - V(s))

Where:
- V(s) = current state value estimate
- G = actual observed return
- alpha = learning rate

Core Concept:
Monte Carlo learning waits until the entire episode finishes before updating values. It learns using actual final outcomes instead of estimated future values.

Unlike Temporal Difference methods such as SARSA and Q-learning:
- no bootstrapping is used
- no future state estimation occurs during interaction

This makes Monte Carlo learning:
- intuitive
- simple
- based entirely on real observed outcomes

Observations:
- Learning is slower compared to TD methods.
- Updates happen only after episode completion.
- Returns are calculated backward from terminal states.
- Longer episodes delay learning updates.
- More episodes generally improve value estimation accuracy.

Practical Understanding:
Monte Carlo methods are useful for understanding the fundamentals of episodic Reinforcement Learning and return-based value estimation.

This experiment demonstrates how agents can learn state values from repeated complete experiences and discounted future rewards without using estimated future state values during interaction.
