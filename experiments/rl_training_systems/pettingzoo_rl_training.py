from pettingzoo.classic import tictactoe_v3
import random


env = tictactoe_v3.env()

episodes = 20

win_count = 0


print("\n=== PETTINGZOO RL TRAINING ===\n")


for episode in range(episodes):

    env.reset()

    total_reward = 0

    print(f"\nEpisode {episode + 1}")

    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        total_reward += reward

        if termination or truncation:

            action = None

        else:

            action_mask = observation["action_mask"]

            valid_actions = []

            for i in range(len(action_mask)):

                if action_mask[i] == 1:
                    valid_actions.append(i)

            action = random.choice(valid_actions)

        env.step(action)

        print("Agent:", agent)

        print("Reward:", reward)

        print("Action:", action)

        print("----------------------")

    print("Episode Reward:", total_reward)

    if total_reward > 0:
        win_count += 1


accuracy = (win_count / episodes) * 100

print(f"\nPositive Reward Episodes: {accuracy:.2f}%")
