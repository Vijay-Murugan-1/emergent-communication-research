Q-Learning Grid Agent

Objective

To understand how an RL agent learns optimal actions using Q-learning and rewards.

---

What the Experiment Does

- Agent moves inside a grid environment.
- Actions:
  - LEFT
  - RIGHT
- Goal state gives positive reward.
- Other movements give negative reward.
- Q-values update after every action.

Over multiple episodes:

- useful actions gain higher Q-values,
- policy improves gradually.

---

Observations

- Initial behavior is random.
- Q-values increase for better actions.
- Agent gradually prefers efficient paths.
- Rewards shape future decision making.

---

Main Understanding

The agent learns optimal behavior by updating Q-values using rewards and future reward estimates.
