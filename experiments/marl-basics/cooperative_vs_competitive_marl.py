# =========================================================
# WEEK 2 - DAY 2
# COOPERATIVE VS COMPETITIVE MARL CONCEPTS
# BASIC INTUITION EXPERIMENTS
# =========================================================


# Cooperative Reward System

print("\n==============================")
print("COOPERATIVE REWARD SYSTEM")
print("==============================\n")

agentA_action = "push_box"
agentB_action = "push_box"

if agentA_action == "push_box" and agentB_action == "push_box":
    team_reward = 10
    print("Both agents cooperated successfully!")
else:
    team_reward = -5
    print("Cooperation failed!")

print("Shared Team Reward:", team_reward)


# Competitive Reward System

print("\n==============================")
print("COMPETITIVE REWARD SYSTEM")
print("==============================\n")

agentA_score = 8
agentB_score = 5

if agentA_score > agentB_score:
    rewardA = 10
    rewardB = -10
    winner = "Agent A"

elif agentB_score > agentA_score:
    rewardA = -10
    rewardB = 10
    winner = "Agent B"

else:
    rewardA = 0
    rewardB = 0
    winner = "DRAW"

print("Winner:", winner)
print("Agent A Reward:", rewardA)
print("Agent B Reward:", rewardB)


# Shared Reward vs Individual Reward Comparison

print("\n==============================")
print("SHARED VS INDIVIDUAL REWARDS")
print("==============================\n")

# Shared reward case
shared_reward = 10

print("Shared Reward Scenario")
print("Agent A Reward:", shared_reward)
print("Agent B Reward:", shared_reward)

print("\nIndividual Reward Scenario")

individual_reward_A = 10
individual_reward_B = 2

print("Agent A Reward:", individual_reward_A)
print("Agent B Reward:", individual_reward_B)


# Coordination Challenge Simulation

print("\n==============================")
print("COORDINATION CHALLENGE")
print("==============================\n")

agentA_path = "LEFT"
agentB_path = "LEFT"

if agentA_path == agentB_path:
    print("Collision occurred!")
    reward = -5
else:
    print("Successful coordination!")
    reward = 10

print("Reward:", reward)


# Independent vs Collaborative Learning Intuition

print("\n==============================")
print("INDEPENDENT VS COLLABORATIVE LEARNING")
print("==============================\n")

independent_agent_reward = 5
collaborative_team_reward = 15

print("Independent Learning Reward:", independent_agent_reward)
print("Collaborative Team Reward:", collaborative_team_reward)

if collaborative_team_reward > independent_agent_reward:
    print("Collaboration produced better overall outcome.")
else:
    print("Independent learning performed better.")
