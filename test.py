import torch
import torch.nn as nn

a = torch.tensor([[1., 2., 3.],
                  [4., 5., 5]], requires_grad=True)
b = torch.rand([6, 3],requires_grad=True)
c = a.repeat(2, 1)

e = torch.cat((a, c), dim=0)
linear_drop = nn.Dropout(p=0.2)

f = b * e
f = linear_drop(f)
h = f.mean()
h.backward()

print(a.grad)
print(b.grad)
print(c.grad)
print(e.grad)

