from environments.two_symbol_comm_env import TwoSymbolCommEnv

env = TwoSymbolCommEnv()

observation, info = env.reset()

print("Observation:", observation)

action = env.action_space.sample()

print("Action Chosen:", action)

observation, reward, terminated, truncated, info = env.step(action)

print("Reward:", reward)
print(env.message_log)
print(env.episode_log)
print(env.metrics_log)
env.render()