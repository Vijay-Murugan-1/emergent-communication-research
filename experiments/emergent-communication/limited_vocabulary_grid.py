import random


episodes = 30

stable_success = 0
random_success = 0


print("\n=== COMMUNICATION EFFICIENCY COMPARISON ===\n")


# Stable Communication
stable_policy = {
    "LEFT": 0,
    "RIGHT": 1
}


for episode in range(episodes):

    goal = random.choice(["LEFT", "RIGHT"])

    message = stable_policy[goal]

    if message == 0:
        action = "LEFT"
    else:
        action = "RIGHT"

    if action == goal:
        stable_success += 1


# Random Communication
for episode in range(episodes):

    goal = random.choice(["LEFT", "RIGHT"])

    message = random.choice([0, 1])

    if message == 0:
        action = "LEFT"
    else:
        action = "RIGHT"

    if action == goal:
        random_success += 1


print("Stable Communication Success:", stable_success)

print("Random Communication Success:", random_success)
