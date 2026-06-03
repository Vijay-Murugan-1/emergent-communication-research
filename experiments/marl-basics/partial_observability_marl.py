# =========================================================
# LOCAL OBSERVATION SIMULATION
# =========================================================

print("\n==============================")
print("LOCAL OBSERVATION SIMULATION")
print("==============================\n")

global_environment = {
    "left_zone": "fire",
    "center_zone": "safe",
    "right_zone": "survivor"
}

agentA_observation = {
    "left_zone": global_environment["left_zone"]
}

agentB_observation = {
    "right_zone": global_environment["right_zone"]
}

print("Global Environment:")
print(global_environment)

print("\nAgent A Observation:")
print(agentA_observation)

print("\nAgent B Observation:")
print(agentB_observation)


# =========================================================
# INFORMATION ASYMMETRY AND COMMUNICATION
# =========================================================

print("\n==============================")
print("INFORMATION ASYMMETRY AND COMMUNICATION")
print("==============================\n")

goal_location = "RIGHT"

agentA_information = goal_location

agentB_decision = "LEFT"

print("Agent B Initial Decision:", agentB_decision)

message = agentA_information

print("\nAgent A sends message:", message)

agentB_decision = message

print("Agent B Updated Decision:", agentB_decision)

if agentB_decision == goal_location:
    print("\nSuccessful Coordination!")
    print("Reward: +10")
else:
    print("\nCoordination Failed!")


# =========================================================
# CENTRALIZED VS DECENTRALIZED OBSERVATION
# =========================================================

print("\n==============================")
print("CENTRALIZED VS DECENTRALIZED OBSERVATION")
print("==============================\n")

global_state = {
    "enemy_position": "LEFT",
    "resource_position": "RIGHT",
    "safe_zone": "CENTER"
}

centralized_view = global_state

agent1_view = {
    "enemy_position": global_state["enemy_position"]
}

agent2_view = {
    "resource_position": global_state["resource_position"]
}

print("Centralized View:")
print(centralized_view)

print("\nAgent 1 Local View:")
print(agent1_view)

print("\nAgent 2 Local View:")
print(agent2_view)


# =========================================================
# COORDINATION UNDER UNCERTAINTY
# =========================================================

print("\n==============================")
print("COORDINATION UNDER UNCERTAINTY")
print("==============================\n")

agentA_path = "LEFT"
agentB_unknown_path = "LEFT"

print("Agent A Path:", agentA_path)
print("Agent B Path:", agentB_unknown_path)

if agentA_path == agentB_unknown_path:
    print("\nCollision Occurred!")
    reward = -5
else:
    print("\nSuccessful Coordination!")
    reward = 10

print("Reward:", reward)
