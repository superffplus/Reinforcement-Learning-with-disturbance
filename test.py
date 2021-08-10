import torch
from agent import Agent
import torch.autograd as autograd


model = Agent(6, 1, None)
x = torch.rand([4, 6])
y = model.value_model(x)
aux = torch.rand([4, 1], requires_grad=True)
result = (aux * y).mean()
loss = autograd.grad(result, model.value_model.parameters())
for learning_step in [0.,] + [.5 ** j for j in range(10)]:
    print(learning_step)
