%pip install gymnasium 
%pip install matplotlib 
%pip install numpy 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env=gym.make("CartPole-v1",render_mode="human")

observation,info=env.reset()
sum=0
for _ in range(200):
  action=env.action_space.sample() #specifies random action

  observation, reward, terminated, truncated, info = env.step(action)
  sum+=reward
  print("Observation: ",observation)
  print("Reward: ",reward)
  if terminated or truncated:
    observation,info=env.reset()
print("Sum: ",sum)
env.close()
