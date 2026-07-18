# Speaker-Listener Communication Environment

## Overview

This environment is a clean PettingZoo AEC (Agent-Environment-Cycle) multi-agent setup for studying cooperative emergent communication.

A speaker observes a hidden goal and communicates a message using a symbolic channel. A listener receives the message and performs an action. Both agents are rewarded based on whether the listener's action matches the goal.

## Agents

Exactly two agents:
- `speaker`
- `listener`

## Action Space

The action spaces are Gymnasium Discrete spaces:
- `speaker`: `Discrete(2)` -> Symbol/message symbol (0 or 1) sent over the channel.
- `listener`: `Discrete(2)` -> Direction choice (0 for LEFT, 1 for RIGHT).

## Observation Space

The observation spaces are Gymnasium Discrete spaces:
- `speaker`: `Discrete(2)` -> The hidden goal representation (0 for LEFT, 1 for RIGHT).
- `listener`: `Discrete(2)` -> The received message symbol (0 or 1) from the communication channel.

## Rewards

- **Cooperative**: Both agents receive the same reward at the end of each step/episode.
- `+10` if the listener chooses the direction matching the speaker's hidden goal.
- `-10` if the listener chooses the wrong direction.

## Episode Cycles

1. `speaker` is selected. It observes the hidden goal and takes a `step(message)` to transmit a message.
2. `listener` is selected. It observes the message from the communication channel and takes a `step(direction)` to choose a direction.
3. The episode terminates, and rewards (+10 or -10) are distributed to both agents.