import sys
# Add virtual environment site-packages if running globally
sys.path.append(".venv/lib/python3.12/site-packages")

from environments.speaker_listener import SpeakerListenerEnv

# 1. Initialize environment
env = SpeakerListenerEnv()

# 2. Reset with a seed for deterministic goal choice
print("=== Resetting environment ===")
env.reset(seed=42)

print("Possible agents:", env.possible_agents)
print("Active agents:", env.agents)

# Iterate through AEC cycle manually
while env.agents:
    current_agent = env.agent_selection
    observation = env.observe(current_agent)
    
    # Check current state/accumulated rewards
    reward = env._cumulative_rewards[current_agent]
    termination = env.terminations[current_agent]
    truncation = env.truncations[current_agent]
    info = env.infos[current_agent]
    
    print(f"\nAgent Selection: {current_agent}")
    print(f"Observation: {observation}")
    print(f"Accumulated Reward: {reward}")
    print(f"Termination: {termination}, Truncation: {truncation}")
    
    if termination or truncation:
        # Step with None when agent is dead
        print(f"Stepping dead agent {current_agent} with action=None")
        env.step(None)
    else:
        # Sample random action
        action = env.action_space(current_agent).sample()
        print(f"Stepping active agent {current_agent} with action={action}")
        env.step(action)
        
        # Render state after speaker transmits or listener takes action
        env.render()

print("\nActive agents remaining after cycle:", env.agents)
