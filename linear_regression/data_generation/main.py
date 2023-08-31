import numpy as np
import matplotlib.pyplot as plt
import torch

# np.random.seed(2525)
# torch.manual_seed((2525))
# # x = np.random.rand(100, 1)
# x = torch.rand(100, 1)
# y = 2 + 3 * x + .1 * torch.randn(100, 1)

# plt.plot(x.numpy(), y.numpy(), '.')
# plt.show()

# torch.set_printoptions(precision=2)
# z = torch.cat([x, y], dim=1)
# print(z)
# print(len(z))

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 + 3 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# plot the data
fig = plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1, title='train')
plt.plot(x_train, y_train, '.r')
plt.subplot(2, 1, 2, title='val')
plt.plot(x_val, y_val, '.')
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

lr = 1e-1
n_epochs = 1000

torch.manual_seed(1411)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    loss.backward()
    print(a.grad)
    print(b.grad)

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    a.grad.zero_()
    b.grad.zero_()

print(a, b)
