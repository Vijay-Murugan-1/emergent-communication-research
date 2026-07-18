"""
Logging and event management for the Speaker-Listener environment.
"""

from typing import List, Dict, Any
from .events import create_event
from .constants import SPEAKER


class EnvironmentLogger:
    """
    Manages logging of messages, episodes, and structured events.
    """

    def __init__(self) -> None:
        self.message_log: List[int] = []
        self.episode_log: List[Dict[str, Any]] = []
        self.events_log: List[Dict[str, Any]] = []

    def log_message(self, message: int) -> None:
        """
        Log a message symbol and emit a structured event.
        """
        self.message_log.append(message)
        
        # Emit structured message event
        event = create_event("message_transmitted", sender=SPEAKER, message=message)
        self.events_log.append(event)

    def log_episode(self, data: Dict[str, Any]) -> None:
        """
        Log episode results and emit a structured event.
        """
        self.episode_log.append(data)
        
        # Emit structured episode event
        event = create_event(
            "episode_completed",
            goal=data.get("goal"),
            message=data.get("message"),
            listener_action=data.get("listener_action"),
            reward=data.get("reward")
        )
        self.events_log.append(event)

    def reset(self) -> None:
        """
        Reset all logs and events.
        """
        self.message_log.clear()
        self.episode_log.clear()
        self.events_log.clear()