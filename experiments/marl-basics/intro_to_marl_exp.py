# Cooperative Gridworld Intuition

print("\n==============================")
print("COOPERATIVE GRIDWORLD INTUITION")
print("==============================\n")

agentA_position = 0
agentB_position = 0
goal_position = 5

steps = 0

while True:

    agentA_position += 1
    agentB_position += 1

    steps += 1

    print(f"Step {steps}")
    print(f"Agent A Position: {agentA_position}")
    print(f"Agent B Position: {agentB_position}")
    print("--------------------------")

    if agentA_position == goal_position and agentB_position == goal_position:
        print("Both agents coordinated successfully!")
        print("Shared Reward: +10")
        break


# Shared Reward System

print("\n==============================")
print("SHARED REWARD SYSTEM")
print("==============================\n")

agentA_action = "move_box"
agentB_action = "move_box"

if agentA_action == "move_box" and agentB_action == "move_box":
    reward = 10
    print("Task completed successfully!")
else:
    reward = -5
    print("Coordination failed!")

print("Shared Reward:", reward)


# Information Asymmetry and Communication Intuition

print("\n==============================")
print("COMMUNICATION INTUITION")
print("==============================\n")

target_location = "LEFT"

agentA_information = target_location

agentB_guess = "RIGHT"

print("Actual Target:", target_location)
print("Agent B Initial Guess:", agentB_guess)

message = agentA_information

print("\nAgent A sends message:", message)

agentB_guess = message

print("Agent B Updated Direction:", agentB_guess)

if agentB_guess == target_location:
    print("Coordination successful!")
    print("Reward: +10")
else:
    print("Failed coordination!")
