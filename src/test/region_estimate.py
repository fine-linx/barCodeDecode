import os

import cv2 as cv
import torch
from PIL import Image
from torchvision import transforms

from resnet.CustomResNet import CustomResNet
from src.BarCodeDecoder import BarCodeDecoder


class RegionEstimator:
    def __init__(self, _re_model=None):
        self.re_model = _re_model
        self._create_preprocess()
        self.offset = 5
        self.barCodeDecoder = BarCodeDecoder()

    def set_re_model(self, _re_model):
        self.re_model = _re_model

    def _create_preprocess(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def estimate(self, source: str, save_cropped=False, save_rect=False, decoder="zbar"):
        assert self.re_model is not None
        image = cv.imread(source)
        input_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        input_image = self.preprocess(input_image)
        input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            self.re_model.eval()
            output = self.re_model(input_image)
        value1, value2, value3, value4 = output[0]
        width, height = image.shape[:2]
        x1 = int(width * value1 - 0.5 * width * value3 - self.offset)
        x2 = int(width * value1 + 0.5 * width * value3 + self.offset)
        y1 = int(height * value2 - 0.5 * height * value4 - self.offset)
        y2 = int(height * value2 + 0.5 * height * value4 + self.offset)
        cropped_img = image[y1:y2, x1:x2]
        file_name = source.split("/")[-1]
        if save_rect:
            img_rect = cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            save_dir = source.replace(file_name, "rect/")
            os.makedirs(save_dir, exist_ok=True)
            cv.imwrite(save_dir + file_name, img_rect, [cv.IMWRITE_PNG_COMPRESSION, 0])
        if save_cropped:
            save_dir = source.replace(file_name, "cropped/")
            os.makedirs(save_dir, exist_ok=True)
            cv.imwrite(save_dir + file_name, cropped_img, [cv.IMWRITE_PNG_COMPRESSION, 0])
        result = self.barCodeDecoder.decode([cropped_img], decoder=decoder, rotate=False)
        return result, cropped_img


if __name__ == '__main__':
    re = RegionEstimator()
    re_model = CustomResNet()
    re_model.load_state_dict(torch.load("../../resnet/checkpoints/adam_best_v1.pt", map_location="cpu"))
    re.set_re_model(re_model)
    folder = "../../db/20231024/unresolved/halcon/rotated/"
    files = os.listdir(folder)
    all_count = 0
    right_count = 0
    for file in files:
        if file.endswith(".png"):
            all_count += 1
            res, _ = re.estimate(folder + file, save_rect=True, save_cropped=True, decoder="halcon")
            if len(res) > 0:
                right_count += 1
            print(file, end="\t")
            print(res)
    print("all: ", all_count)
    print("right: ", right_count)
    print("acc: ", right_count / all_count if all_count > 0 else 0)
