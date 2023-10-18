import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BarCode(Dataset):
    def __init__(self, root_dir, _transforms=None):
        self.root_dir = root_dir
        self.transforms = _transforms
        self.image_list = os.listdir(os.path.join(root_dir, "images"))
        # self.label_list = os.listdir(os.path.join(root_dir, "labels"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "images", self.image_list[idx])
        label_name = os.path.join(self.root_dir, "labels", self.image_list[idx].replace(".jpg", ".txt"))
        image = Image.open(img_name)
        if image.mode != "RGB":
            image = image.convert("RGB")
        with open(label_name, 'r') as label_file:
            label_data = label_file.readline().strip().split()
            label = list(map(float, label_data[1:]))
            label = torch.from_numpy(np.array(label)).float()
        if self.transforms:
            image = self.transforms(image)
        return image, label


if __name__ == '__main__':
    root = "dataset/train"
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_data = BarCode(root, preprocess)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    for images, labels in train_dataloader:
        print(images.shape)
        print(labels.shape)
        # print(labels)
        break
