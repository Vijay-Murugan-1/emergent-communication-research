Monte Carlo State Value Estimation

Objective

Implement Monte Carlo learning for state value estimation using the FrozenLake-v1 environment.

---

Key Concepts

- Episodic learning
- Monte Carlo returns
- Discounted rewards
- State value estimation
- Learning from complete outcomes

---

Monte Carlo Return Formula

G_t = R_(t+1) + γR_(t+2) + γ²R_(t+3) + ...

Where:

- "G_t" = total discounted return
- "γ" = discount factor
- "R" = future rewards

---

State Value Update Equation

V(s) = V(s) + α [ G - V(s) ]

Where:

- "V(s)" = current state value
- "G" = actual observed return
- "α" = learning rate

---

Core Idea

Monte Carlo learning waits until the entire episode finishes before updating values.

Learning uses:

- actual observed returns
- complete episode outcomes

No future estimation is used during interaction.

---

Environment

Environment used:

- FrozenLake-v1
- "is_slippery=False"

The agent learns state values from repeated episodes.

---

Important Process

1. Run complete episode
2. Store state-reward history
3. Calculate returns backward
4. Update state values

---

Characteristics

Monte Carlo learning:

- is intuitive
- uses real final outcomes
- learns after episode completion
- does not bootstrap future estimates

---

Observations

- Learning is slower than TD methods.
- Longer episodes delay updates.
- More episodes improve estimation accuracy.
- State values gradually converge with repeated experience.

---

Difference from TD Learning

Monte Carlo:

- waits until episode ends
- uses actual final returns

TD Learning:

- updates during interaction
- uses estimated future values
