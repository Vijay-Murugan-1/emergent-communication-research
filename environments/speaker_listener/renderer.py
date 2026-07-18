"""
Renderer for the Speaker-Listener environment.
"""

from .state import EnvironmentState
from .communication import CommunicationChannel


def render_env(state: EnvironmentState, channel: CommunicationChannel) -> None:
    """
    Render the current environment state to the console.
    """
    print(f"\n--- Environment Render ---")
    print(f"Step Count: {state.step_count}")
    print(f"Goal: {state.goal}")
    print(f"Current Agent: {state.current_agent}")
    print(f"Channel Message: {channel.receive()}")
    print(f"--------------------------\n")
