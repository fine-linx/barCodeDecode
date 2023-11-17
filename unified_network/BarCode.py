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
        self.image_list = os.listdir(self.root_dir)
        # self.label_list = os.listdir(os.path.join(root_dir, "labels"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        label = self.image_list[idx].split("_")[0]
        regression_value_path = os.path.join(self.root_dir[:self.root_dir.rfind("/")], "labels",
                                             self.image_list[idx].split(".")[0] + ".txt")
        image = Image.open(img_name)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        label = [int(digit) for digit in label]
        label = torch.tensor(label)
        with open(regression_value_path, 'r') as label_file:
            label_data = label_file.readline().strip().split()
            regression_value = list(map(float, label_data))
            regression_value = torch.from_numpy(np.array(regression_value)).float()

        return image, label, regression_value


if __name__ == '__main__':
    root = "E:/work/barCode/net_dataset3"
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_data = BarCode(root, preprocess)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    for images, labels, regression_values in train_dataloader:
        print(images.shape)
        print(labels.shape)
        print(regression_values.shape)
        print(labels)
        print(regression_values)
        # break
