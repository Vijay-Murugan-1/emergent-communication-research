import gymnasium as gym
env=gym.make("FrozenLake-v1",render_mode="human",is_slippery=False) 
onservation,info=env.reset()
for stepm in range(70):
  action=env.action_space.sample() 
  observation,reward,terminated,truncated,info=env.step(action)

  print("State: ",observation)
  print("Reward: ",reward)
  
  if terminated or truncated:
    print("Episode ended")

    observation,info=env.reset()
env.close()
