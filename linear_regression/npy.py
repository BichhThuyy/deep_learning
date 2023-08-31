import npy as np

np.random.seed(123)
n_train = 100
x_train = np.random.rand(n_train, 1)
y_train = 2 + 3 * x_train + .1 * np.random.randn(n_train, 1)

a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    yhat = a + b * x_train
    error = (y_train - yhat)
    loss = (error ** 2).mean()

    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)
