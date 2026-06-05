# Trainable Sender–Receiver Communication

## Objective

To understand how agents can improve communication using rewards instead of hardcoded message meanings.

---

## What the Experiment Does

- Sender observes the goal.
- Sender sends a message.
- Receiver interprets the message.
- Receiver performs an action.
- Environment gives reward based on coordination success.

## If communication fails:

- sender updates its message policy.

Over multiple episodes:

- successful communication patterns stabilize.

---

## Main Concepts

### Sender–Receiver System

One agent sends information while another agent uses it to complete the task.

---

### Reward-Based Communication

Useful communication survives because it produces higher rewards.

---

### Adaptive Messaging

Message meanings are not manually fixed and can change during learning.

---

### Emergent Communication Intuition

Agents gradually learn which messages improve coordination.

---

## Observations

- Initial communication is inconsistent.
- Failed communication causes policy updates.
- Successful communication patterns become stable.
- Coordination accuracy improves over time.

---
## Limitations of This Experiment
This is still a simplified communication-learning system.
The experiment does not yet include:
neural networks,
deep reinforcement learning,
gradient optimization,
trainable policies,
or actual MARL training frameworks.
However, it introduces the core intuition behind:
emergent communication systems

---
## Important Understanding

This experiment demonstrates the basic idea behind emergent communication:

- communication evolves through reward optimization instead of manual design.
