import random


# Environment settings
goals = ["left", "right"]
symbols = ["0", "1"]

episodes = 50


# Initial communication policy
message_policy = {
    "left": random.choice(symbols),
    "right": random.choice(symbols)
}


success_count = 0


print("Two Symbol Communication System")


for episode in range(episodes):

    # Sender observes goal
    goal = random.choice(goals)

    # Sender sends symbol
    message = message_policy[goal]

    # Receiver interprets symbol
    if message == "0":
        action = "left"
    else:
        action = "right"


    # Reward
    if action == goal:
        reward = 1
        success_count += 1
    else:
        reward = 0


    # Update communication mapping if failed
    if reward == 0:
        message_policy[goal] = random.choice(symbols)


    print(
        "Episode:",
        episode + 1,
        "Goal:",
        goal,
        "Message:",
        message,
        "Action:",
        action,
        "Reward:",
        reward
    )


print(
    "Success rate:",
    success_count / episodes
)