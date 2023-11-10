import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class DecodeNet(nn.Module):
    def __init__(self, num_classes=10, num_digits=13):
        super(DecodeNet, self).__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        self._create_resnet(num_classes, num_digits)

    def _create_resnet(self, num_classes, num_digits):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # resnet18 = models.resnet18()
        # resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512 * 1),
            nn.Linear(512 * 1, num_classes * num_digits))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = x.view(x.size(0), self.num_digits, -1)
        return x


if __name__ == '__main__':
    model = DecodeNet()
    print(model)
    # 1. 准备输入图像
    image_path = "E:/work/barCode/net_dataset/8032987643521_0.png"  # 替换成您的图像路径
    image = Image.open(image_path)

    # 定义图像预处理转换，确保与模型的输入匹配
    preprocess = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 对图像应用预处理
    input_image = preprocess(image)
    # print(input_image)
    input_image = input_image.unsqueeze(0)  # 添加批次维度，将其变成形状为(1, 3, 224, 224)的张量
    criterion = nn.CrossEntropyLoss()
    y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3])
    model.eval()
    with torch.no_grad():
        output = model(input_image)
    output = output.view(-1, 10)
    output = nn.functional.softmax(output, dim=-1)
    print(output)
    loss = criterion(output, y)
    print("loss: ", loss.item())
