import gymnasium as gym
import time

# Create Taxi environment
env = gym.make("Taxi-v4", render_mode="human")

# Reset environment
state, info = env.reset()

total_reward = 0

for step in range(50):

    # Random action
    action = env.action_space.sample()

    # Perform action
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep: {step}")
    print(f"Current State: {state}")
    print(f"Action Taken: {action}")
    print(f"Next State: {next_state}")
    print(f"Reward: {reward}")

    total_reward += reward

    # Update current state
    state = next_state

    time.sleep(0.5)

    # Episode ends
    if terminated or truncated:
        print("\nEpisode Ended")

        # Restart environment
        state, info = env.reset()

print("\nTotal Reward:", total_reward)

env.close()
