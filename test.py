import pandas as pd
import torch

a = [1, 2, 3, 4]
b = [4, 5, 6, 1]
c = [1, 2, 6, 6]

d = pd.DataFrame([a, b, c], columns=['a', 'b', 'c', 'd'])
print(d)
d['a'][2] = 7
print(d)
t = torch.tensor(d.values, dtype=torch.float32)
print(t)
