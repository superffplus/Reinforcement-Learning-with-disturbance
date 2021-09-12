import pandas as pd
import torch
import numpy as np


a = torch.rand([64, 1])
b = torch.rand([64, 3])
c = torch.rand([64, 1])
print((a / b) * c)
