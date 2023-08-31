import torch
import torch.optim as optim
import torch.nn as nn

n_data = 100
x_train = torch.rand(n_data, 1)
y_train = 2 + 3 * x_train + .1 * torch.randn(n_data, 1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train.to(device)
y_train.to(device)

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

lr = 1e-1
n_epochs = 1000
optimizer = optim.SGD([a, b], lr=lr)
loss_fn = nn.MSELoss(reduction='mean')

for epoch in range(n_epochs):
    yhat = a + b * x_train

    loss = loss_fn(y_train, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(a, b)
