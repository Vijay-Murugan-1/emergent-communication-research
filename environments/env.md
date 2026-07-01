
# Two Symbol Communication Environment

## Overview

This environment is a simple Speaker-Listener communication setup for studying emergent communication.

A speaker observes a goal and communicates using a symbolic message. A listener interprets the message and selects an action. The environment provides a reward based on whether the listener's action matches the goal.

## Agents

### Speaker

The speaker generates a symbolic message based on the given goal/observation.

### Listener

The listener receives the symbolic message and selects an action based on that message.

## Action Space

The environment uses a discrete action space:

- `0` → Symbol/message representing LEFT
- `1` → Symbol/message representing RIGHT

The speaker's action is treated as the communicated message.

## Observation Space

The observation space is discrete:

- `0` → LEFT goal
- `1` → RIGHT goal

## Action Space

The environment uses a discrete action space:

- `0` → Symbol/message representing LEFT
- `1` → Symbol/message representing RIGHT

The speaker's action is treated as the communicated message.

## Observation Space

The observation space is discrete:

- `0` → LEFT goal
- `1` → RIGHT goal

## Logging

The environment maintains lightweight logs for evaluation:

- `message_log` → Stores communication symbols/messages sent by the speaker.
- `episode_log` → Stores complete episode interactions including goal, message, action, and reward.
- `metrics_log` → Stores evaluation metrics such as success and reward.

## Current Limitations

- The current environment uses a simple two-symbol communication system.
- The listener behaviour is directly determined by the received symbol.
- Learning-based speaker and listener agents can be integrated in future versions.