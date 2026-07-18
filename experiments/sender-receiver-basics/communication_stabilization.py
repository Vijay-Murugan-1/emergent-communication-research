import random


message_policy = {
    "LEFT": random.choice([0, 1]),
    "RIGHT": random.choice([0, 1])
}

episodes = 100

success_count = 0


print("\n=== COMMUNICATION STABILIZATION ===\n")


for episode in range(episodes):

    goal = random.choice(["LEFT", "RIGHT"])

    message = message_policy[goal]

    if message == 0:
        action = "LEFT"
    else:
        action = "RIGHT"

    if action == goal:

        reward = 10
        success_count += 1

    else:

        reward = -10

        message_policy[goal] = random.choice([0, 1])

    if (episode + 1) % 10 == 0:

        accuracy = (success_count / (episode + 1)) * 100

        print(f"Episode: {episode + 1}")

        print("Policy:", message_policy)

        print(f"Accuracy: {accuracy:.2f}%")

        print("----------------------")


print("\nFinal Policy:", message_policy)

final_accuracy = (success_count / episodes) * 100

print(f"Final Accuracy: {final_accuracy:.2f}%")
