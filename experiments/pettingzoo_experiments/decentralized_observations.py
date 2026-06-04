from pettingzoo.classic import tictactoe_v3

env = tictactoe_v3.env()

env.reset()

for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()

    print("\nCurrent Agent:", agent)

    print("Observation Shape:")
    print(observation["observation"].shape)

    print("Action Mask:")
    print(observation["action_mask"])

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)
