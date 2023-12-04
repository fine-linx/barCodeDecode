import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.utils import get_key_by_value, FIRST_DIGIT_MAP, ENCODE_MAP


class BarCode(Dataset):
    def __init__(self, root_dir, _transforms=None):
        self.root_dir = root_dir
        self.transforms = _transforms
        self.image_list = os.listdir(root_dir)
        # self.image_list = [file for file in os.listdir(self.root_dir)
        #                    if os.path.isfile(os.path.join(self.root_dir, file))]
        # self.label_list = os.listdir(os.path.join(root_dir, "labels"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        label = self.image_list[idx].split("_")[0]
        image = Image.open(img_name)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        label = [int(digit) for digit in label]
        label = torch.tensor(label)
        return image, label


class BarCodeBinary(BarCode):
    def __init__(self, root_dir, _transforms=None):
        super(BarCodeBinary, self).__init__(root_dir, _transforms)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        label = self.image_list[idx].split("_")[0]
        image = Image.open(img_name)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        label = self._get_label(label)
        return image, label

    @staticmethod
    def _get_label(label_str: str) -> torch.Tensor:
        left_format = get_key_by_value(int(label_str[0]), FIRST_DIGIT_MAP)
        label_binary_str = ''.join([
            ENCODE_MAP[digit][0] if ch == "O" else ENCODE_MAP[digit][1]
            for ch, digit in zip(left_format, label_str[1:7])
        ])
        label_binary_str += ''.join([ENCODE_MAP[digit][2] for digit in label_str[7:]])

        label = [float(digit) for digit in label_binary_str]
        label = torch.tensor(label)
        return label


if __name__ == '__main__':
    root = "E:/work/barCode/net_dataset4/train_sub/"
    preprocess = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_data = BarCodeBinary(root, preprocess)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    for images, labels in train_dataloader:
        print(images.shape)
        print(labels.shape)
        print(labels)
        # break
