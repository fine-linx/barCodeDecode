import os
import shutil
import time

import torch
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from BarCode import BarCode
from DetNet import DetNet


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
        # transforms.Normalize(mean=[0.173, 0.173, 0.173], std=[0.254, 0.254, 0.254])  # 训练集的数据
        # 验证集数据
        # mean: tensor([0.1742, 0.1742, 0.1742]), std: tensor([0.2561, 0.2561, 0.2561])
    ])
    batch_size = 32
    root = "E:/work/barCode/net_dataset3/"
    out_dir = "tune/"
    prefix = "resnet50_v0.2_"
    os.makedirs(out_dir, exist_ok=True)
    train_data = BarCode(root_dir=root + "train", _transforms=preprocess)
    valid_data = BarCode(root_dir=root + "valid", _transforms=preprocess)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    model = DetNet()
    model.load_state_dict(torch.load("tune/resnet50_v0.1_adam_best.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 超参
    criterionCEL = nn.CrossEntropyLoss()
    criterionMSE = nn.MSELoss(reduction='mean')
    gamma = 0.5
    learning_rate = 1e-5
    momentum = 0.9
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    epochs = 300

    early_stop = 30
    best_count = 0
    best_epoch = 0
    max_acc = 0.0
    for epoch in range(epochs):
        if best_count >= early_stop:
            print("early stop")
            break
        print(f"\nEpoch:{epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}")
        model.train()
        sum_loss = 0.0
        regression_sum_loss, classification_sum_loss = 0.0, 0.0
        # 训练
        for batch_idx, (images, labels, regression_values) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels, regression_values = images.to(device), labels.to(device), regression_values.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # 切分为两个部分，第一个部分为4个值的回归，第二个部分为130个值的分类,
            regression_output, classification_output = outputs.split([4, 130], dim=1)

            classification_output = classification_output.reshape(-1, 10)
            classification_output = nn.functional.softmax(classification_output, dim=-1)
            labels = labels.view(-1)
            classification_loss = criterionCEL(classification_output, labels)

            # regression_output = nn.ReLU()(regression_output)
            regression_output = regression_output.squeeze()
            regression_loss = criterionMSE(regression_output, regression_values)

            loss = classification_loss + regression_loss * gamma
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            regression_sum_loss += regression_loss.item()
            classification_sum_loss += classification_loss.item()
            print(f"[epoch: {epoch + 1}, iter: {batch_idx + 1 + epoch * length}] Loss: {sum_loss / (batch_idx + 1)}, "
                  f"regression loss: {regression_sum_loss / (batch_idx + 1)}, "
                  f"classification loss: {classification_sum_loss / (batch_idx + 1)}")

        # 验证
        print("waiting val...")
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(valid_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, outputs = outputs.split([4, 130], dim=1)
                outputs = outputs.reshape(-1, 13, 10)
                outputs = nn.functional.softmax(outputs, dim=-1)
                _, predicted = torch.max(outputs, 2)
                are_equal = torch.eq(predicted, labels)
                all_equal = torch.all(are_equal, dim=1)
                correct = all_equal.sum().item()
                total_correct += correct
                total_samples += labels.size(0)
            accuracy = total_correct / total_samples
            print(f"Epoch [{epoch + 1}/{epochs}]: Validation Accuracy = {accuracy}")
        torch.save(model.state_dict(), out_dir + prefix + "adam_last.pt")
        best_count += 1
        if accuracy > max_acc:
            best_count = 0
            torch.save(model.state_dict(), out_dir + prefix + "adam_best.pt")
            print(f"save epoch {epoch + 1} model")
            best_epoch = epoch
            max_acc = accuracy
        scheduler.step()
    print(f"best epoch: {best_epoch + 1}, accuracy: {max_acc}")


def predict():
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = DetNet()
    model.load_state_dict(torch.load("tune/resnet50_v0.2_adam_best.pt"))
    model.eval()

    folder = "E:/work/barCode/net_dataset3/valid/"
    unresolved_folder = folder + "unresolved/network/"
    os.makedirs(unresolved_folder, exist_ok=True)
    files = os.listdir(folder)

    all_count = 0
    maybe_right_count = 0
    right_count = 0
    wrong_list = list()
    t1 = time.time()
    for file in files:
        if file.endswith(".png"):
            print(folder + file, end="\t")
            all_count += 1
            img = Image.open(folder + file)
            input_img = preprocess(img)
            input_img = input_img.unsqueeze(0)
            with torch.no_grad():
                output = model(input_img)
            _, output = output.split([4, 130], dim=1)
            output = output.reshape(-1, 13, 10)
            output = nn.functional.softmax(output, dim=-1)
            # result = modified_predict(output)
            _, predicted = torch.max(output, 2)
            arr = predicted.squeeze().numpy()
            result = "".join(map(str, arr))
            if is_valid_ean13(result):
                maybe_right_count += 1
                print("maybe right", end="\t")
                label = file.split("_")[0]
                if label == result:
                    right_count += 1
                else:
                    wrong_list.append(folder + file)
                    shutil.copy(folder + file, unresolved_folder + file)
            else:
                wrong_list.append(folder + file)
                shutil.copy(folder + file, unresolved_folder + file)
            print(result)
        # if all_count > 5000:
        #     break
    print("wrong:")
    for file in wrong_list:
        print(file)
    print(f"all: {all_count}")
    print(f"maybe right: {maybe_right_count}")
    print(f"right: {right_count}")
    print("acc: ", right_count / all_count if all_count > 0 else 0)
    t2 = time.time()
    print("total time: %s ms, per image: %s ms" % ((t2 - t1) * 1000, (t2 - t1) * 1000 / all_count))


def modified_predict(logit: torch.Tensor) -> str:
    logit = logit.squeeze()
    top2_list = []
    diff_list = []
    for group in logit:
        top_indices = torch.topk(group, k=2).indices.numpy()
        # 获取group在这两个位置的差值
        diff = group[top_indices[0]] - group[top_indices[1]]
        diff_list.append(diff.item())
        top2_list.append(top_indices)
    # 获取diff_list最小值的索引
    # max_index = np.argmin(diff_list)
    # 一位容错
    candidate = [digit[0] for digit in top2_list]
    _candidate = "".join(map(str, candidate))
    if is_valid_ean13(_candidate):
        return _candidate
    for i in range(len(top2_list)):
        data = candidate.copy()
        data[i] = top2_list[i][1]
        _candidate = "".join(map(str, data))
        if is_valid_ean13(_candidate):
            return _candidate

    # 对每一组进行组合，得到所有可能得结果
    # candidates = list(product(*top2_list))
    # for candidate in candidates:
    #     result = "".join(map(str, candidate))
    #     if is_valid_ean13(result):
    #         return result
    return ""


if __name__ == '__main__':
    # main()
    predict()
