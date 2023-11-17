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

folder = "E:/work/barCode/net_dataset3/"
label_folder = folder + "label/"
rect_folder = folder + "rect/"
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
        label_file = file.split(".")[0] + ".txt"
        with open(label_folder + label_file, "w") as f:
            f.write(" ".join(map(str, output)))
        value1, value2, value3, value4 = output
        img_cv = cv.imread(folder + file)
        width, height = img_cv.shape[:2]
        x1 = max(width * value1 - 0.5 * width * value3 - offset, 0)
        x2 = min(width * value1 + 0.5 * width * value3 + offset, width)
        y1 = max(height * value2 - 0.5 * height * value4 - offset, 0)
        y2 = min(height * value2 + 0.5 * height * value4 + offset, height)
        img_cv = cv.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.imwrite(folder + "rect/" + file, img_cv, [cv.IMWRITE_PNG_COMPRESSION, 0])
