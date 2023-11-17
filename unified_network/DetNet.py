import os

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


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


class DetNet(nn.Module):
    def __init__(self, num_classes=10, num_digits=13):
        self.inplanes = 1024
        super(DetNet, self).__init__()
        self.num_classes = num_classes
        self.num_digits = num_digits
        self._create_detnet(num_classes, num_digits)

    def _create_detnet(self, num_classes, num_digits):
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # resnet = models.resnet18()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        self.det_layer = self._make_layers()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(self.inplanes, num_classes * num_digits + 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes * num_digits + 4),
        )

    def _make_layers(self):
        layers = list()
        layers.append(BottleneckB(self.inplanes, self.inplanes // 4))
        for i in range(2):
            layers.append(BottleneckA(self.inplanes, self.inplanes // 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.resnet(x)
        out = self.det_layer(out)
        out = self.pooling(out)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    import cv2 as cv

    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = nn.Sequential(*list(model.children())[:-3])
    model = DetNet()
    print(model)
    # print(list(model.children()))
    # model.load_state_dict(torch.load("tune/adam_best.pt"))
    model.eval()
    # print(model)
    # 定义图像预处理转换，确保与模型的输入匹配
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    folder = "E:/work/barCode/net_dataset3/"
    label_folder = folder + "label/"
    rect_folder = folder + "temp/"
    os.makedirs(rect_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    files = os.listdir(folder)
    offset = 5
    for file in files:
        if file.endswith(".png"):
            img = Image.open(folder + file)
            input_img = preprocess(img)
            input_img = input_img.unsqueeze(0)
            with torch.no_grad():
                output = model(input_img)
            output = output.squeeze().tolist()
            # label_file = file.split(".")[0] + ".txt"
            # with open(label_folder + label_file, "w") as f:
            #     f.write(" ".join(map(str, output)))
            value1, value2, value3, value4 = output[:4]
            img_cv = cv.imread(folder + file)
            width, height = img_cv.shape[:2]
            x1 = max(width * value1 - 0.5 * width * value3 - offset, 0)
            x2 = min(width * value1 + 0.5 * width * value3 + offset, width)
            y1 = max(height * value2 - 0.5 * height * value4 - offset, 0)
            y2 = min(height * value2 + 0.5 * height * value4 + offset, height)
            img_cv = cv.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv.imwrite(rect_folder + file, img_cv, [cv.IMWRITE_PNG_COMPRESSION, 0])
