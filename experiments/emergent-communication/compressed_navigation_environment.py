import random


GRID_SIZE = 5

TARGETS = [0, 1, 2, 3, 4]

episodes = 30

success_count = 0


print("\n=== COMPRESSED NAVIGATION ENVIRONMENT ===\n")


for episode in range(episodes):

    target = random.choice(TARGETS)

    receiver_position = random.randint(0, GRID_SIZE - 1)

    print(f"\nEpisode {episode + 1}")

    print("Target:", target)

    print("Initial Position:", receiver_position)


    # Bottleneck communication
    if target <= 2:
        message = 0
    else:
        message = 1


    print("Message:", message)


    # Receiver action
    if message == 0:
        receiver_position -= 1
    else:
        receiver_position += 1


    receiver_position = max(
        0,
        min(receiver_position, GRID_SIZE - 1)
    )


    print("New Position:", receiver_position)


    # Reward
    if receiver_position == target:

        reward = 10
        success_count += 1

        print("Target Reached!")

    else:

        reward = -1

        print("Target Missed!")


    print("Reward:", reward)

    print("----------------------")


accuracy = (success_count / episodes) * 100

print(f"\nFinal Accuracy: {accuracy:.2f}%")
