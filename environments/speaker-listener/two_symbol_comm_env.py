"""
Speaker-Listener communication environment for EC-Lab.

This environment simulates communication between two agents:
- Speaker: sends a message based on the observation.
- Listener: receives the message and performs an action.

The goal is to study emergent communication behaviour.
"""

import gymnasium as gym 
from gymnasium import spaces 
import random


class TwoSymbolCommEnv(gym.Env):
    """
    A simple two-symbol Speaker-Listener communication environment.

    The environment contains:
    - A hidden goal: LEFT or RIGHT.
    - A speaker that sends a symbolic message through an action.
    - A listener that interprets the message and selects a direction.

    Reward:
    +10  if the listener chooses the correct goal.
    -10  if the listener chooses the wrong goal.
    """
    def __init__(self) ->None :
      super().__init__()
      self.action_space=spaces.Discrete(2)
      self.observation_space=spaces.Discrete(2)
      self.goals=["LEFT","RIGHT"]
      ##self.message_policy={"LEFT": random.choice([0,1]),"RIGHT": random.choice([0,1])}
      self.message_log = []
      self.episode_log = []
      self.metrics_log = []
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        self.goal=random.choice(self.goals)
        if self.goal == "LEFT" :
           observation = 0
        else:
           observation = 1
        return observation,{}
    
    def step(self, action: int) -> tuple[int, int, bool, bool, dict]:
         message =int(action)
         self.message_log.append(message)
         if message == 0:
            receiver_action = "LEFT"
         else:
            receiver_action = "RIGHT"
         if receiver_action == self.goal:
            reward = 10
         else:
            reward = -10
        
         terminated = True
         truncated = False

         observation = 0 if self.goal == "LEFT" else 1

         info = {}
         self.episode_log.append(
             {
                 "goal": self.goal,
                 "message": message,
                 "receiver_action": receiver_action,
                 "reward": reward
             }
         )
         self.metrics_log.append(
             {
                 "success": reward == 10,
                 "reward": reward
             }
         )

         return observation, reward, terminated, truncated, info
      
    def render(self)-> None:
         print(f"Goal: {self.goal}")
    def close(self) -> None:
        """
        Clean up resources used by the environment.
        """
        pass