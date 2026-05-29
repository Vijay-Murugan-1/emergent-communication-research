import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)

state_values = np.zeros(env.observation_space.n)

alpha = 0.1
gamma = 0.9
episodes = 5000

for episode in range(episodes):

    episode_data = []

    state, _ = env.reset()

    done = False

    while not done:

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        episode_data.append((state, reward))

        state = next_state

    # Monte Carlo Return Calculation
    G = 0

    for state, reward in reversed(episode_data):

        G = reward + gamma * G

        state_values[state] = state_values[state] + alpha * (
            G - state_values[state]
        )

print("State Values:")
print(state_values)
