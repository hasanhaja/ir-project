import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

a = y * y * 3
out = a.mean()

print(a, out)

z = torch.randn(2, 2)
z = ((z * 3) / (z - 1))
print(z.requires_grad)
print(z)

z.requires_grad_(True)
print(z.requires_grad)

u = (z * z).sum()
print(u.grad_fn)
print(u)

print("Back prop...")
print(out)

out.backward()

print(x.grad)