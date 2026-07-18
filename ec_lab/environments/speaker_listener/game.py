"""
Core game logic.
"""

from .reward import compute_reward


class SpeakerListenerGame:
    """
    Handles the environment game logic.
    """

    def __init__(self, state, communication):
        self.state = state
        self.communication = communication

    def speaker_step(self, message: int):
        """
        Speaker sends a message.
        """
        self.communication.transmit(message)

    def listener_step(self, action: str):
        """
        Listener performs an action.
        """
        reward = compute_reward(
            self.state.goal,
            action,
        )

        return reward