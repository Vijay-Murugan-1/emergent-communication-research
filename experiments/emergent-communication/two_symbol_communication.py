import random


GOALS = ["LEFT", "RIGHT"]

SYMBOLS = [0, 1]

episodes = 50


# Initial random communication policy
message_policy = {
    "LEFT": random.choice(SYMBOLS),
    "RIGHT": random.choice(SYMBOLS)
}


success_count = 0


print("\n=== TWO SYMBOL COMMUNICATION SYSTEM ===\n")


for episode in range(episodes):

    goal = random.choice(GOALS)

    # Sender sends message
    message = message_policy[goal]

    # Receiver interprets symbol
    if message == 0:
        action = "LEFT"
    else:
        action = "RIGHT"

    # Reward
    if action == goal:

        reward = 10
        success_count += 1

    else:

        reward = -10

        # Change communication mapping
        message_policy[goal] = random.choice(SYMBOLS)

    print(f"Episode {episode + 1}")

    print("Goal:", goal)

    print("Message:", message)

    print("Action:", action)

    print("Reward:", reward)

    print("Policy:", message_policy)

    print("------------------------")


accuracy = (success_count / episodes) * 100

print(f"\nFinal Accuracy: {accuracy:.2f}%")
