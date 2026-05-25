import numpy as np
import random


# Reward probabilities for 4 slot machines
bandits = [0.2, 0.5, 0.75, 0.3]

# Tracking rewards
estimated_rewards = [0, 0, 0, 0]
action_counts = [0, 0, 0, 0]

epsilon = 0.2
total_reward = 0

for step in range(100):

    # Exploration vs Exploitation
    if random.random() < epsilon:
        action = random.randint(0, 3)
        print("Exploring...")
    else:
        action = np.argmax(estimated_rewards)
        print("Exploiting...")

    # Generate reward
    if random.random() < bandits[action]:
        reward = 1
    else:
        reward = 0

    total_reward += reward

    # Update estimates
    action_counts[action] += 1
    # Uses Incremental Action Value Update algorithm (Qnew = Qold + alpha*(R-Qold))
    # where alpha = 1/N
    estimated_rewards[action] += (  
        reward - estimated_rewards[action]
    ) / action_counts[action]

    print(f"Step: {step}")
    print(f"Chosen Bandit: {action}")
    print(f"Reward: {reward}")
    print(f"Estimated Rewards: {estimated_rewards}")
    print("------------------------")

print("Total Reward:", total_reward)
