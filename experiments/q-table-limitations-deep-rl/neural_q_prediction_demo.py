import numpy as np

state = np.array([0.5, 0.2, 0.8])

weights = np.array([
    [0.2, 0.4, 0.1],
    [0.5, 0.3, 0.7],
    [0.6, 0.9, 0.2]
])

q_values = np.dot(state, weights)

print(q_values)
