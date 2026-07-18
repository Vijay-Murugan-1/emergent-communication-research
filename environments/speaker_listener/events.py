"""
Structured events for replay and visualization.
"""


def create_event(event_type: str, **payload):
    """
    Create a structured event.
    """

    return {
        "event": event_type,
        "payload": payload,
    }