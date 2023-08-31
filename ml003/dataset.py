import numpy as np
import torch
import glob
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

batch_size = 100


def check_img(img_path):
    try:
        img_read = Image.open(img_path)
        if img_read is not None:
            return True
        else:
            return False
    except:
        return False


dogs = glob.glob(os.path.join('.\data\Dog', '*.jpg'))
cats = glob.glob(os.path.join('.\data\Cat', '*.jpg'))
print(len(cats), len(dogs))

dog_array = list(filter(check_img, dogs))
cat_array = list(filter(check_img, cats))
print(len(cat_array), len(dog_array))

dog_list = [[img, 1] for img in dog_array]
cat_list = [[img, 0] for img in cat_array]
# print(cat_list.index([r'.\data\Cat\666.jpg', 0]))

dog_train, dog_test = train_test_split(dog_list, test_size=0.2)
cat_train, cat_test = train_test_split(cat_list, test_size=0.2)
#
train_list, val_list = train_test_split(np.concatenate((dog_train, cat_train)), test_size=0.2)
test_list = np.concatenate((dog_test, cat_test))
# print(len(train_list), len(val_list), len(test_list))
# print(train_list[:5])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img = self.file_list[idx]
        img_path = img[0]
        label = img[1]
        img_read = Image.open(img_path)
        img_transformed = self.transform(img_read)
        current_channels, height, width = img_transformed.shape
        if current_channels != 3:
            add_channels = 3 - current_channels
            if add_channels > 0:
                padding = torch.zeros(add_channels, height, width)
                img_transformed = torch.cat((img_transformed, padding), dim=0)
            if add_channels < 0:
                img_transformed = img_transformed[:3, :, :]
        return img_transformed, label, img_path


train_data = dataset(train_list, transform=train_transforms)
val_data = dataset(val_list, transform=val_transforms)
test_data = dataset(test_list, transform=test_transforms)
#
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
