import torch

x=torch.tensor([1,2,3])
y=torch.tensor([4,5,6])

print("Addition: ")
print(x+y)

print("Multiplication: ")
print(x*y)

print("Dot product:")
print(torch.dot(x,y))

matrix=torch.tensor([[1,2],[3,4]])
print("Shape: ")
print(matrix.shape)
