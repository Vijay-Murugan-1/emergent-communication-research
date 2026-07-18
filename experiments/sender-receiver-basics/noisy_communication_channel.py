import random


message_policy = {
    "LEFT": 0,
    "RIGHT": 1
}

episodes = 50

success_count = 0

noise_probability = 0.3


print("\n=== NOISY COMMUNICATION CHANNEL ===\n")


for episode in range(episodes):

    goal = random.choice(["LEFT", "RIGHT"])

    message = message_policy[goal]

    # Add communication noise
    if random.random() < noise_probability:
        message = random.choice([0, 1])

    if message == 0:
        action = "LEFT"
    else:
        action = "RIGHT"

    if action == goal:

        reward = 10
        success_count += 1

    else:
        reward = -10

    print(f"Episode {episode + 1}")

    print("Goal:", goal)

    print("Received Message:", message)

    print("Action:", action)

    print("Reward:", reward)

    print("----------------------")


accuracy = (success_count / episodes) * 100

print(f"\nFinal Accuracy: {accuracy:.2f}%")
