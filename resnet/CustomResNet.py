import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self._create_resnet()

    def _create_resnet(self):
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # resnet18 = models.resnet18()
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        regression_layer = nn.Sequential(nn.Conv2d(512, 4, 1),
                                         nn.ReLU()
                                         )
        self.resnet18.add_module("output", regression_layer)

    def forward(self, x):
        return self.resnet18(x)


if __name__ == '__main__':
    model = CustomResNet()
    # 1. 准备输入图像
    image_path = "../cropped_images/db2/20230206091306472969_S_01_cropped_0.JPG"  # 替换成您的图像路径
    image = Image.open(image_path)

    # 定义图像预处理转换，确保与模型的输入匹配
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 对图像应用预处理
    input_image = preprocess(image)
    print(input_image)
    input_image = input_image.unsqueeze(0)  # 添加批次维度，将其变成形状为(1, 3, 224, 224)的张量
    model.eval()
    with torch.no_grad():
        output = model(input_image)
        value1, value2, value3, value4 = output[0]
        print(f"Value 1: {value1.item()}")
        print(f"Value 2: {value2.item()}")
        print(f"Value 3: {value3.item()}")
        print(f"Value 4: {value4.item()}")
