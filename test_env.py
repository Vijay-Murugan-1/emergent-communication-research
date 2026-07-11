import sys
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 40)
print("Python Environment Test")
print("=" * 40)

print(f"Python Version : {sys.version}")
print(f"Platform       : {platform.platform()}")

# NumPy Test
arr = np.array([1, 2, 3, 4, 5])
print("\nNumPy Test")
print("Array:", arr)
print("Mean :", np.mean(arr))

# Pandas Test
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Score": [85, 90, 95]
})

print("\nPandas Test")
print(df)

# Matplotlib Test
plt.plot([1, 2, 3], [2, 4, 6])
plt.title("Environment Test Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("test_plot.png") # Changed from show() to savefig() to avoid GUI hang in background
print("Plot saved as test_plot.png")

print("\n✅ Environment is working correctly!")
