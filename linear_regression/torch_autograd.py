import torch

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

for epoch in range(n_epochs):
    yhat = a + b * x_train
    error = y_train - yhat
    loss = (error ** 2).mean()

    loss.backward()

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    a.grad.zero_()
    b.grad.zero_()

print(a, b)
