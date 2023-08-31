import torch
import torch.optim as optim
import torch.nn as nn


class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=2)

    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        return self.linear(x)


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


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

# Creates the train_step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer)
losses = []

for epoch in range(n_epochs):
    loss = train_step(x_train, y_train)
    losses.append(loss)

print(model.state_dict())
