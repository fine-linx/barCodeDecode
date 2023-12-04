import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from decode_network.BarCode import BarCode

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_mean_std_value(loader):
    data_sum, data_squared_sum, num_batches = 0, 0, 0
    print(len(loader))
    for data, _ in loader:
        data_sum += torch.mean(data, dim=[0, 2, 3])
        data_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
        print(num_batches)
    mean = data_sum / num_batches
    std = (data_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


if __name__ == '__main__':
    root = "E:/work/barCode/net_dataset4/"
    root2 = "E:/work/barCode/net_dataset5/"
    train_data = BarCode(root_dir=root + "train", _transforms=preprocess)
    valid_data = BarCode(root_dir=root + "valid", _transforms=preprocess)
    train_data2 = BarCode(root_dir=root2 + "train", _transforms=preprocess)
    valid_data2 = BarCode(root_dir=root2 + "valid", _transforms=preprocess)
    # train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    # valid_data_loader = DataLoader(valid_data, batch_size=128, shuffle=True)
    data_loader = DataLoader(train_data + valid_data + train_data2 + valid_data2, batch_size=128, shuffle=True)
    mean, std = get_mean_std_value(data_loader)
    print(f"mean: {mean}, std: {std}")
