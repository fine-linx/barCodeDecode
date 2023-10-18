import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from BarCode import BarCode
from CustomResNet import CustomResNet


def main():
    # 定义图像预处理转换，确保与模型的输入匹配
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 32
    root = "dataset/"
    out_dir = "checkpoints/"
    train_data = BarCode(root_dir=root + "train", _transforms=preprocess)
    valid_data = BarCode(root_dir=root + "valid", _transforms=preprocess)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    model = CustomResNet()
    model.load_state_dict(torch.load("checkpoints/adam_best_v1.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 超参
    criterion = nn.MSELoss(reduction='mean')
    learning_rate = 1e-5
    momentum = 0.9
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 300

    loss_list = []
    best_epoch = 0
    for epoch in range(epochs):
        print(f"\nEpoch:{epoch + 1}")
        model.train()
        sum_loss = 0.0
        # 训练
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            print(f"[epoch: {epoch + 1}, iter: {batch_idx + 1 + epoch * length}] Loss: {sum_loss / (batch_idx + 1)}")

        # 验证
        print("waiting val...")
        with torch.no_grad():
            loss_total = 0.0
            for batch_idx, (images, labels) in enumerate(valid_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels)
                loss_total += loss.item()
            print(f"val loss: {loss_total}")
            loss_list.append(loss_total)
        torch.save(model.state_dict(), out_dir + "adam_last.pt")
        if loss_total <= min(loss_list):
            torch.save(model.state_dict(), out_dir + "adam_best.pt")
            print(f"save epoch {epoch + 1} model")
            best_epoch = epoch
    print(f"best epoch: {best_epoch + 1}")


if __name__ == '__main__':
    main()
