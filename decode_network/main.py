import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from BarCode import BarCode
from DecodeNet import DecodeNet


# from src.util.util import is_valid_ean13


def is_valid_ean13(barcode):
    # 确保输入是一个13位的数字字符串
    if not barcode.isdigit() or len(barcode) != 13:
        return False

    # 计算校验位
    odd_sum = sum(int(barcode[i]) for i in range(0, 12, 2))
    even_sum = sum(int(barcode[i]) for i in range(1, 12, 2))
    total = odd_sum + even_sum * 3
    checksum = (10 - (total % 10)) % 10

    # 检查校验位是否与计算的校验位相符
    return int(barcode[12]) == checksum


def main():
    # 定义图像预处理转换，确保与模型的输入匹配
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 32
    root = "E:/work/barCode/net_dataset2/"
    out_dir = "checkpoints/"
    train_data = BarCode(root_dir=root + "train", _transforms=preprocess)
    valid_data = BarCode(root_dir=root + "valid", _transforms=preprocess)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    model = DecodeNet()
    model.load_state_dict(torch.load("checkpoints/adam_best_v3.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 超参
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 200

    early_stop = 20
    best_count = 0
    best_epoch = 0
    max_acc = 0.0
    for epoch in range(epochs):
        if best_count >= early_stop:
            print("early stop")
            break
        print(f"\nEpoch:{epoch + 1}")
        model.train()
        sum_loss = 0.0
        # 训练
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1, 10)
            outputs = nn.functional.softmax(outputs, dim=-1)
            labels = labels.view(-1)
            # loss_per_digit = criterion(outputs, labels)
            # loss_per_digit = loss_per_digit.view(-1, 13)
            # loss = loss_per_digit.mean(dim=1)
            # loss = loss.mean()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            print(f"[epoch: {epoch + 1}, iter: {batch_idx + 1 + epoch * length}] Loss: {sum_loss / (batch_idx + 1)}")

        # 验证
        print("waiting val...")
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.view(-1, 13, 10)
                outputs = nn.functional.softmax(outputs, dim=-1)
                _, predicted = torch.max(outputs, 2)
                are_equal = torch.eq(predicted, labels)
                all_equal = torch.all(are_equal, dim=1)
                correct = all_equal.sum().item()
                total_correct += correct
                total_samples += labels.size(0)
            accuracy = total_correct / total_samples
            print(f"Epoch [{epoch + 1}/{epochs}]: Validation Accuracy = {accuracy}")
        torch.save(model.state_dict(), out_dir + "adam_last.pt")
        best_count += 1
        if accuracy > max_acc:
            best_count = 0
            torch.save(model.state_dict(), out_dir + "adam_best.pt")
            print(f"save epoch {epoch + 1} model")
            best_epoch = epoch
            max_acc = accuracy
    print(f"best epoch: {best_epoch + 1}, accuracy: {max_acc}")


def predict():
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = DecodeNet()
    model.load_state_dict(torch.load("checkpoints/adam_best.pt"))
    model.eval()

    folder = "E:/work/barCode/20231026/folder_1/rotated/"
    files = os.listdir(folder)

    all_count = 0
    maybe_right_count = 0
    for file in files:
        if file.endswith(".png"):
            print(file, end="\t")
            all_count += 1
            img = Image.open(folder + file)
            input_img = preprocess(img)
            input_img = input_img.unsqueeze(0)
            with torch.no_grad():
                output = model(input_img)
            output = output.view(-1, 13, 10)
            output = nn.functional.softmax(output, dim=-1)
            _, predicted = torch.max(output, 2)
            arr = predicted.squeeze().numpy()
            result = "".join(map(str, arr))
            if is_valid_ean13(result):
                maybe_right_count += 1
                print("maybe right", end="\t")
            print(result)
    print(f"all: {all_count}")
    print(f"maybe right: {maybe_right_count}")
    print("acc: ", maybe_right_count / all_count if all_count > 0 else 0)


if __name__ == '__main__':
    # main()
    predict()
