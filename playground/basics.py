import torch
import numpy as np

x = torch.empty(5, 3)
print("Uninitialized will have whatever values were in that memory location at the time:")
print(x)

x = torch.rand(5, 3)
print("Randomly initialized:")
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print("Zeros: ")
print(x)

x = torch.tensor([5.5, 3])
print("Directly from data: ")
print(x.size())

x = torch.ones(5, 3, dtype=torch.int)
print(x)

y = torch.ones(5, 3, dtype=torch.int)
# You can transpose tensors like you can with numpy
# y = y.t
# print(y)
print(x + y)

# convert numpy arrays to tensors

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# And to put things in the GPU
# The `torch.device` objects can help move things in and out the GPU

if torch.cuda.is_available():
    print("GPU available!!")
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))

print("Done!")