import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader
from cnn_model import Cnn

lr = 0.001
batch_size = 100
epochs = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device == 'cuda':
    torch.cuda.manual_seed_all(1234)

# DYRECTORY = '.\data'
# CATEGORY = ['Cat', 'Dog']
# for category in CATEGORY:
#     folder = os.path.join(DYRECTORY, category)
#     print(folder)
#     for img in os.listdir(folder):
#         img_path = os.path.join(folder, img)
#         print(img_path)
#         break

# random_idx = np.random.randint(1, len(dog_list), size=10)
# fig = plt.figure()
# i = 1
# for idx in random_idx:
#     print(dog_list[idx])
#     ax = fig.add_subplot(2, 5, i)
#     img = Image.open(dog_list[idx])
#     plt.imshow(img)
#     i += 1
#
# plt.axis('off')
# plt.show()


# print(len(train_loader), len(val_loader), len(test_loader))


model = Cnn().to(device)
model.train()

optimizer = optim.Adam(params=model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = torch.tensor([int(lab) for lab in label]).to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = torch.tensor([int(lab) for lab in label]).to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_val_accuracy, epoch_val_loss))

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
