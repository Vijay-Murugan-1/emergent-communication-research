import random


GRID_SIZE = 5

ACTIONS = ["LEFT", "RIGHT"]

Q_table = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 100


def get_q(state, action):

    if (state, action) not in Q_table:
        Q_table[(state, action)] = 0

    return Q_table[(state, action)]


print("\n=== Q LEARNING GRID AGENT ===\n")


for episode in range(episodes):

    state = random.randint(0, GRID_SIZE - 1)

    goal = GRID_SIZE - 1

    while state != goal:

        if random.random() < epsilon:

            action = random.choice(ACTIONS)

        else:

            left_q = get_q(state, "LEFT")
            right_q = get_q(state, "RIGHT")

            if right_q >= left_q:
                action = "RIGHT"
            else:
                action = "LEFT"

        if action == "RIGHT":
            next_state = min(state + 1, GRID_SIZE - 1)

        else:
            next_state = max(state - 1, 0)

        if next_state == goal:
            reward = 10
        else:
            reward = -1

        old_q = get_q(state, action)

        next_max = max(
            get_q(next_state, "LEFT"),
            get_q(next_state, "RIGHT")
        )

        new_q = old_q + alpha * (
            reward + gamma * next_max - old_q
        )

        Q_table[(state, action)] = new_q

        state = next_state


print("\nLearned Q Table:\n")

for key, value in Q_table.items():
    print(key, ":", round(value, 2))
