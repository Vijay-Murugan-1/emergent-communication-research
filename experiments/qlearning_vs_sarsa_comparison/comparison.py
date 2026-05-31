import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = env.observation_space.n
action_size = env.action_space.n

# Separate Q-Tables
q_learning_table = np.zeros((state_size, action_size))
sarsa_table = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 50000

# -------------------------------
# Q-LEARNING
# -------------------------------

for episode in range(episodes):

    state, _ = env.reset()
    done = False

    while not done:

        # Epsilon-greedy action
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_learning_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Q-Learning Update
        q_learning_table[state, action] = (
            q_learning_table[state, action]
            + alpha * (
                reward
                + gamma * np.max(q_learning_table[next_state])
                - q_learning_table[state, action]
            )
        )

        state = next_state

# -------------------------------
# SARSA
# -------------------------------

for episode in range(episodes):

    state, _ = env.reset()

    # Initial action
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(sarsa_table[state])

    done = False

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Next action
        if random.uniform(0,1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(sarsa_table[next_state])

        # SARSA Update
        sarsa_table[state, action] = (
            sarsa_table[state, action]
            + alpha * (
                reward
                + gamma * sarsa_table[next_state, next_action]
                - sarsa_table[state, action]
            )
        )

        state = next_state
        action = next_action

# -------------------------------
# OUTPUT
# -------------------------------

print("Q-Learning Q-Table:")
print(q_learning_table)

print("\nSARSA Q-Table:")
print(sarsa_table)
