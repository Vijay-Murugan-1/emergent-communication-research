This experiment implements the SARSA (State-Action-Reward-State-Action) algorithm on the FrozenLake-v1 environment using tabular Reinforcement Learning.

SARSA is an on-policy Temporal Difference (TD) learning algorithm where the agent updates Q-values using the actual next action selected by the current policy. Unlike Q-learning, which updates using the maximum possible future Q-value, SARSA updates using the Q-value of the next action actually chosen by the agent through the epsilon-greedy policy.

The main objective of this experiment is to understand:
- On-policy learning
- Temporal Difference learning
- Exploration-aware learning
- Q-table updates
- Policy-based learning behavior

Environment:
FrozenLake-v1 is a grid-world environment where the agent must move from the start state to the goal state while avoiding holes. In this implementation, deterministic movement is used with:
is_slippery=False

Q-Table:
The Q-table stores state-action values:
Q(state, action)

Each Q-value represents the estimated long-term usefulness of taking a specific action from a particular state.

Action Selection:
The agent follows an epsilon-greedy policy:
- Most of the time, the agent selects the action with the highest Q-value.
- Sometimes, random exploration occurs to discover new paths and state transitions.

SARSA Update Rule:

Q(s,a) = Q(s,a) + alpha * [reward + gamma * Q(s',a') - Q(s,a)]

Where:
- s = current state
- a = current action
- s' = next state
- a' = actual next action selected by policy
- alpha = learning rate
- gamma = discount factor

Core Concept:
The most important idea behind SARSA is that learning updates use the actual next action selected by the current policy. This means exploration directly affects learning updates.

If the policy selects a weaker exploratory action in the next state, SARSA still learns from that action instead of assuming optimal future behavior.

This makes SARSA:
- exploration-aware
- more conservative
- safer in risky environments

On-Policy Learning:
SARSA is called an on-policy algorithm because:
- the policy used for acting
and
- the policy used for learning
are the same.

The agent learns based on the exact behavior policy currently being followed.

Observations:
- Q-values improve gradually over episodes.
- Exploration strongly affects learning behavior.
- Different runs can produce different Q-tables due to stochastic exploration.
- Higher episode counts generally improve convergence.
- Hyperparameters like epsilon, alpha, and gamma significantly affect learning speed and stability.

Practical Understanding:
SARSA is useful in environments where exploration risk matters because it learns considering actual future behavior rather than ideal optimal behavior.

This experiment demonstrates how reinforcement learning agents gradually improve policies through repeated interaction, reward feedback, and Temporal Difference updates.
