from pettingzoo.classic import tictactoe_v3

env = tictactoe_v3.env()

env.reset()

print("Agents in Environment:")
print(env.agents)

print("\nStarting Agent Iteration:\n")

for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()

    print("Current Agent:", agent)
    print("Reward:", reward)
    print("Termination:", termination)

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    print("Chosen Action:", action)
    print("--------------------------")

    env.step(action)
