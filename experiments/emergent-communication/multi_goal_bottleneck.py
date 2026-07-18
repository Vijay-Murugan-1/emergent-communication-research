import random


GOALS = ["LEFT", "RIGHT", "UP", "DOWN"]

episodes = 50

success_count = 0


print("\n=== MULTI GOAL BOTTLENECK SYSTEM ===\n")


for episode in range(episodes):

    goal = random.choice(GOALS)

    # Sender bottleneck communication
    if goal in ["LEFT", "UP"]:
        message = 0
    else:
        message = 1


    # Receiver interpretation
    if message == 0:
        action = random.choice(["LEFT", "UP"])
    else:
        action = random.choice(["RIGHT", "DOWN"])


    # Reward
    if action == goal:

        reward = 10
        success_count += 1

    else:
        reward = -10


    print(f"Episode {episode + 1}")

    print("Goal:", goal)

    print("Message:", message)

    print("Action:", action)

    print("Reward:", reward)

    print("----------------------")


accuracy = (success_count / episodes) * 100

print(f"\nFinal Accuracy: {accuracy:.2f}%")
