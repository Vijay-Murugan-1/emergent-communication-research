import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

episodes = 5000

for episode in range(episodes):

    state, _ = env.reset()

    # Choose initial action
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    done = False

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Choose NEXT action using same policy
        if random.uniform(0,1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state])

        # SARSA Update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward +
            gamma * q_table[next_state, next_action] -
            q_table[state, action]
        )

        state = next_state
        action = next_action

print("Final Q-Table:")
print(q_table)
