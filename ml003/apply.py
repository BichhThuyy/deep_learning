import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from dataset import test_loader
from cnn_model import Cnn
from PIL import Image

print('test_loader length: ', len(test_loader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)

model = Cnn().to(device)
model.load_state_dict(torch.load('model.pth'))

dog_probs = []
model.eval()
with torch.no_grad():
    for data, label, img_path in test_loader:
        # print('data', data)
        # print('fileid', label)
        # print('img_path', img_path)
        # break
        data = data.to(device)
        preds = model(data)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(img_path), preds_list))

# dog_probs.sort(key=lambda x: int(x[0]))
print('dog_probs', dog_probs[:5])

fig = plt.figure()
random_idx = np.random.randint(1, len(dog_probs), size=10)
i = 1
for idx in random_idx:
    img_path = dog_probs[idx][0]
    label = 'Cat' if dog_probs[idx][1] <= 0.5 else 'Dog'
    ax = fig.add_subplot(2, 5, i)
    img = Image.open(img_path)
    ax.set_title(label)
    plt.imshow(img)
    i += 1

plt.axis('off')
plt.show()
