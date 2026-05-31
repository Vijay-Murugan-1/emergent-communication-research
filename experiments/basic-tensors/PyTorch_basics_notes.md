# PyTorch Tensor Basics for Deep Reinforcement Learning

## Introduction

This document contains the foundational PyTorch and tensor concepts studied during the first week of Reinforcement Learning preparation.

The main purpose of learning PyTorch basics at this stage is not to master Deep Learning immediately, but to understand the computational foundation used later in:

* Deep Reinforcement Learning (Deep RL)
* Deep Q Networks (DQN)
* Multi-Agent Reinforcement Learning (MARL)
* Neural Network based AI systems

This serves as the transition from:

* classical tabular Reinforcement Learning
  to
* neural network based Deep RL systems.

---

# What is PyTorch?

PyTorch is an open-source Deep Learning framework developed primarily for:

* neural networks
* tensor computations
* machine learning research
* artificial intelligence systems

PyTorch is widely used in:

* Reinforcement Learning
* Computer Vision
* Natural Language Processing
* Robotics
* Generative AI
* Research systems

In Deep Reinforcement Learning, PyTorch is commonly used to:

* build neural networks,
* represent states as tensors,
* predict Q-values,
* and train intelligent agents.

---

# Why PyTorch is Important in Reinforcement Learning

In classical Reinforcement Learning:

* Q-values are stored explicitly in Q-tables.

Example:

Q(state, action)

This works only for small environments.

Large environments such as:

* robotics,
* autonomous driving,
* Atari games,
* multi-agent systems,
* image-based environments

cannot be solved efficiently using Q-tables because the state space becomes enormous.

Deep Reinforcement Learning solves this by replacing:

* explicit table storage

with:

* neural network based function approximation.

PyTorch provides the tools required for:

* tensor computation,
* neural network creation,
* automatic differentiation,
* and Deep RL training.

---

# What is a Tensor?

A tensor is the fundamental data structure used in PyTorch.

A tensor can be understood as:

* an advanced numerical array,
* a generalized mathematical container,
* or a multidimensional matrix.

Tensors are used to represent:

* states,
* rewards,
* actions,
* observations,
* neural network inputs,
* and outputs.

In Deep Reinforcement Learning:

* almost everything eventually becomes a tensor.

---

# Tensor Dimensions

Tensors can have multiple dimensions.

---

# 0-D Tensor (Scalar)

A single value.

Example:

```python
import torch

x = torch.tensor(5)

print(x)
```

Output:

```python
tensor(5)
```

Represents:

* one numerical value.

---

# 1-D Tensor (Vector)

A list or array of values.

Example:

```python
x = torch.tensor([1,2,3])
```

Output:

```python
tensor([1,2,3])
```

Represents:

* vectors,
* feature lists,
* state values,
* numerical sequences.

---

# 2-D Tensor (Matrix)

Rows and columns.

Example:

```python
x = torch.tensor([
    [1,2],
    [3,4]
])
```

Represents:

* matrices,
* tables,
* batches of data.

---

# Higher-Dimensional Tensors

Deep Learning systems commonly use:

* 3D tensors,
* 4D tensors,
* and even larger dimensional representations.

Examples:

* image batches,
* video frames,
* multi-agent observations.

---

# Tensor Creation

Basic tensor creation:

```python
import torch

x = torch.tensor([1,2,3])

print(x)
```

Output:

```python
tensor([1, 2, 3])
```

This creates a tensor object inside PyTorch.

---

# Tensor Operations

PyTorch tensors support efficient mathematical operations.

---

# Tensor Addition

```python
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

print(x + y)
```

Output:

```python
tensor([5,7,9])
```

Each element is added individually.

---

# Tensor Multiplication

```python
print(x * y)
```

Output:

```python
tensor([4,10,18])
```

Element-wise multiplication occurs.

---

# Dot Product

```python
print(torch.dot(x,y))
```

Calculation:

1×4 + 2×5 + 3×6

Result:

```python
tensor(32)
```

---

# Why Dot Product is Important

Dot products and matrix multiplications form the mathematical foundation of:

* neural networks,
* Deep RL,
* Q-value prediction,
* and function approximation.

Neural networks heavily rely on:

* tensor multiplication,
* weighted computations,
* and matrix operations.

---

# Tensor Shapes

Tensor shape represents:

* dimensions,
* rows,
* columns,
* structural size of tensors.

Example:

```python
x = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print(x.shape)
```

Output:

```python
torch.Size([2,3])
```

Meaning:

* 2 rows
* 3 columns

---

# Why Shapes Matter

In Deep Learning:

* inputs,
* outputs,
* weights,
* and neural layers

must have compatible dimensions.

Incorrect tensor shapes produce computation errors.

Tensor shape understanding becomes extremely important later in:

* Deep Q Networks (DQN)
* Convolutional Neural Networks (CNNs)
* Transformer architectures
* Multi-Agent RL systems

---

# Random Tensors

PyTorch can generate random tensors.

Example:

```python
x = torch.rand((2,3))

print(x)
```

This creates:

* a random 2×3 tensor.

Random initialization is heavily used in:

* neural network weights,
* exploration,
* and training systems.

---

# Neural Network Intuition

At this stage, only a high-level understanding is required.

A neural network fundamentally performs:

Input → Computation → Output

In Deep RL:

* states enter the network,
* weights process information,
* predicted Q-values are produced.

---

# Example Deep RL Intuition

Input state:

```python
[distance, speed, obstacle]
```

The neural network processes this information and predicts:

| Action | Predicted Q-value |
| ------ | ----------------- |
| Left   | 2.1               |
| Right  | 8.4               |
| Jump   | 1.7               |

The agent then selects:

* the action with highest predicted Q-value.

---

# Important Deep RL Understanding

Classical RL:

* explicitly stores Q-values inside tables.

Deep RL:

* predicts Q-values dynamically using learned neural network weights.

This is one of the most important conceptual transitions in modern Reinforcement Learning.

---

# Tensors in Deep Reinforcement Learning

Later in Deep RL, tensors will represent:

* environment states,
* observations,
* rewards,
* action probabilities,
* neural network outputs,
* replay memory batches,
* and communication vectors in MARL.

Examples:

* game frames
* image inputs
* robot sensor values
* agent communication symbols

All are represented numerically as tensors.

---

# Why PyTorch Was Introduced in Week 1

The purpose of introducing PyTorch basics during Week 1 is to:

* prepare for Deep RL,
* understand neural network computation foundations,
* and bridge the gap between:

  * tabular RL
  * and Deep RL systems.

The current goal is NOT:

* advanced Deep Learning mathematics,
* backpropagation derivations,
* or optimization theory.

The goal is only:

* conceptual understanding,
* tensor familiarity,
* and Deep RL preparation.

---

# Key Understanding Summary

## Classical Reinforcement Learning

* Uses Q-tables
* Explicitly stores values
* Works for small environments
* Relies on memorization

---

# Deep Reinforcement Learning

* Uses neural networks
* Uses tensors
* Predicts Q-values dynamically
* Learns generalized patterns
* Scales to large environments

---

# Final Important Understanding

PyTorch provides:

* tensor computation tools,
* neural network frameworks,
* and Deep Learning infrastructure

that later allow Reinforcement Learning agents to:

* approximate Q-values,
* generalize across states,
* and solve complex large-scale environments beyond the capability of tabular Q-learning.
