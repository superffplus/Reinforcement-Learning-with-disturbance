import pandas as pd
import torch
import numpy as np


a = np.random.rand(4, 5)
b = torch.as_tensor(a)
print(b)
