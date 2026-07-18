"""
Communication channel.
"""


class CommunicationChannel:
    """
    Handles message transmission between agents.
    """

    def __init__(self):
        self.message = None

    def transmit(self, message: int):
        self.message = message

    def receive(self):
        return self.message

    def reset(self):
        self.message = None