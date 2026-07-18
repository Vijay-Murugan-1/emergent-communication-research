import random


message_policy = {
    "LEFT": random.choice([0, 1]),
    "RIGHT": random.choice([0, 1])
}

episodes = 50
success_count = 0

print("\n=== TRAINABLE SENDER RECEIVER SYSTEM ===\n")


for episode in range(episodes):

    goal = random.choice(["LEFT", "RIGHT"])

    # Sender
    message = message_policy[goal]

    # Receiver
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

        # Learn new message
        message_policy[goal] = random.choice([0, 1])

    print(f"Episode {episode + 1}")

    print("Goal:", goal)
    print("Message:", message)
    print("Action:", action)
    print("Reward:", reward)

    print("Policy:", message_policy)

    print("----------------------")


print("\nFinal Learned Policy:")
print(message_policy)

accuracy = (success_count / episodes) * 100

print(f"\nSuccess Rate: {accuracy:.2f}%")
