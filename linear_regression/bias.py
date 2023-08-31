import torch
import torch.optim as optim
import torch.nn as nn


class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        self.linear = nn.Linear(1, 1, bias=2)

    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        return self.linear(x)


n_data = 100
x_train = torch.rand(n_data, 1)
y_train = 2 + 3 * x_train + .1 * torch.randn(n_data, 1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train.to(device)
y_train.to(device)

model = LayerLinearRegression().to(device)
# We can also inspect its parameters using its state_dict
print(model.state_dict())

lr = 1e-1
n_epochs = 1000
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train)

    loss = loss_fn(y_train, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())
