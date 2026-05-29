import gymnasium as gym
import numpy as np
import random

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Create Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.99     # discount factor
epsilon = 0.1    # exploration probability

episodes = 1000

for episode in range(episodes):

    # Reset environment
    state, info = env.reset()

    done = False

    while not done:

        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        # Move to next state
        state = next_state

        # Episode ends
        done = terminated or truncated

print("Final Q-Table:\n")
print(q_table)

env.close()
