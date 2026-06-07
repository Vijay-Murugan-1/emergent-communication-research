import random


GRID_SIZE = 5

ACTIONS = ["LEFT", "RIGHT"]

Q_table = {
    (0, "RIGHT"): 5,
    (1, "RIGHT"): 6,
    (2, "RIGHT"): 7,
    (3, "RIGHT"): 8
}


episodes = 20

learned_success = 0
random_success = 0


print("\n=== LEARNED VS RANDOM AGENT ===\n")


# Learned Agent
for episode in range(episodes):

    state = 0
    goal = GRID_SIZE - 1

    steps = 0

    while state != goal and steps < 20:

        right_q = Q_table.get((state, "RIGHT"), 0)
        left_q = Q_table.get((state, "LEFT"), 0)

        if right_q >= left_q:
            action = "RIGHT"
        else:
            action = "LEFT"

        if action == "RIGHT":
            state += 1
        else:
            state -= 1

        state = max(0, min(state, GRID_SIZE - 1))

        steps += 1

    if state == goal:
        learned_success += 1


# Random Agent
for episode in range(episodes):

    state = 0
    goal = GRID_SIZE - 1

    steps = 0

    while state != goal and steps < 20:

        action = random.choice(ACTIONS)

        if action == "RIGHT":
            state += 1
        else:
            state -= 1

        state = max(0, min(state, GRID_SIZE - 1))

        steps += 1

    if state == goal:
        random_success += 1


print("Learned Agent Success:", learned_success)

print("Random Agent Success:", random_success)
