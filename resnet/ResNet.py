import os

import cv2 as cv
import torch
from PIL import Image
from torchvision import transforms

from CustomResNet import CustomResNet

model = CustomResNet()
model.load_state_dict(torch.load("checkpoints/adam_best_v1.pt", map_location="cpu"))
model.eval()

# 定义图像预处理转换，确保与模型的输入匹配
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

folder = "../db/barCodeDB2/rotated/"
files = os.listdir(folder)
for file in files:
    if file.endswith(".png"):
        img = Image.open(folder + file)
        input_img = preprocess(img)
        input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            output = model(input_img)
        value1, value2, value3, value4 = output[0]
        img_cv = cv.imread(folder + file)
        width, height = img_cv.shape[:2]
        x1 = width * value1 - 0.5 * width * value3 - 5
        x2 = width * value1 + 0.5 * width * value3 + 5
        y1 = height * value2 - 0.5 * height * value4 - 5
        y2 = height * value2 + 0.5 * height * value4 + 5
        img_cv = cv.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.imwrite(folder + "rect/" + file, img_cv, [cv.IMWRITE_PNG_COMPRESSION, 0])
