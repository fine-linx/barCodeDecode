from functools import partial

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .GhostNetV2 import ghostnetv2


class BottleneckA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        assert inplanes == (planes * 4), "inplanes != planes * 4"
        assert stride == 1, "stride != 1"
        assert downsample is None, "downsample!= None"
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckB(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        assert inplanes == (planes * 4), "inplanes != planes * 4"
        assert stride == 1, "stride != 1"
        assert downsample is None, "downsample!= None"
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DecodeNet(nn.Module):
    def __init__(self, num_classes=10, num_digits=13, dropout=0.2):
        self.inplanes = 1024
        super(DecodeNet, self).__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.final_drop = dropout
        self._create_resnet(num_classes, num_digits, dropout)

    def _create_resnet(self, num_classes, num_digits, dropout):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        # self.backbone = nn.Sequential(*list(efficientnet.children())[:-2])
        self.den_layer1 = self._make_layers()
        # self.den_layer2 = self._make_layers()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()
        self.fc = nn.Sequential(
            nn.Conv2d(self.inplanes, num_classes * num_digits, 1, bias=False),
            nn.BatchNorm2d(num_classes * num_digits),

        )

    def _make_layers(self):
        layers = list()
        layers.append(BottleneckB(self.inplanes, self.inplanes // 4))
        for i in range(2):
            layers.append(BottleneckA(self.inplanes, self.inplanes // 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.backbone(x)
        out = self.den_layer1(out)
        # out = self.den_layer2(out)
        out = self.pooling(out)
        if self.final_drop > 0:
            out = self.dropout(out)
        out = self.fc(out)
        return out


class DecodeNetBinary(DecodeNet):
    def __init__(self, digit_bits=7, num_digits=12, final_drop=0.2):
        super(DecodeNetBinary, self).__init__(dropout=final_drop)
        self.fc = nn.Sequential(
            nn.Conv2d(self.inplanes, digit_bits * num_digits, 1, bias=False),
            nn.BatchNorm2d(digit_bits * num_digits),
            nn.Sigmoid()
        )


class DecodeNetGhost(nn.Module):
    def __init__(self, digit_bits=7, num_digits=12, final_drop=0.2):
        super(DecodeNetGhost, self).__init__()
        self.inplanes = 960
        self.num_digits = num_digits
        self.digit_bits = digit_bits
        self.final_drop = final_drop
        self._create_ghostnet(digit_bits, num_digits, final_drop)

    def _create_ghostnet(self, digit_bits, num_digits, final_drop):
        ghostNet = partial(ghostnetv2, model_name="1x", final_drop=final_drop, num_classes=digit_bits * num_digits)()
        self.ghostNet = nn.Sequential(*list(ghostNet.children())[:-6])
        self.den_layer1 = self._make_layers()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        if self.final_drop > 0:
            self.dropout = nn.Dropout(self.final_drop)
        else:
            self.dropout = nn.Sequential()
        self.fc = nn.Sequential(
            nn.Conv2d(self.inplanes, digit_bits * num_digits, 1, bias=False),
            nn.BatchNorm2d(digit_bits * num_digits),
            nn.Sigmoid(),
        )

    def _make_layers(self):
        layers = list()
        layers.append(BottleneckB(self.inplanes, self.inplanes // 4))
        for i in range(2):
            layers.append(BottleneckA(self.inplanes, self.inplanes // 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.ghostNet(x)
        out = self.den_layer1(out)
        out = self.pooling(out)
        if self.final_drop > 0:
            out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = DecodeNetBinary()
    # model = models.efficientnet_v2_s()
    # model = models.resnet50()
    # model = models.efficientnet_b0()
    print(model)
    # 1. 准备输入图像
    image_path = "E:/work/barCode/net_dataset/8032987643521_0.png"  # 替换成您的图像路径
    image = Image.open(image_path)

    # 定义图像预处理转换，确保与模型的输入匹配
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
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
    # output = output.view(-1, 10)
    # output = nn.functional.softmax(output, dim=-1)
    print(output)
    print(output.shape)
    # loss = criterion(output, y)
    # print("loss: ", loss.item())
