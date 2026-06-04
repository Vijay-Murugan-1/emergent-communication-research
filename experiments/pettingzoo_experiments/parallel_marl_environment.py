from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.parallel_env()

observations = env.reset()

print("Agents:")
print(env.agents)

print("\nInitial Observations Received\n")

for agent in env.agents:
    print(agent)

actions = {
    agent: env.action_space(agent).sample()
    for agent in env.agents
}

observations, rewards, terminations, truncations, infos = env.step(actions)

print("\nRewards After One Parallel Step:\n")

for agent in rewards:
    print(agent, ":", rewards[agent])
