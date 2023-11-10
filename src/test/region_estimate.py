import os

import cv2 as cv
import halcon
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
        cropped_img = image[y1 - 8:y2 + 8, x1 - 8:x2 + 8]
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
        # result = self.barCodeDecoder.decode([cropped_img], decoder=decoder, rotate=False)
        # result = self.barCodeDecoder._decode(cropped_img, decoder=decoder)
        result = None
        return result, cropped_img


def appendResult(result_dict: dict, data: str):
    if data in result_dict:
        result_dict[data] += 1
    else:
        result_dict[data] = 0


if __name__ == '__main__':
    re = RegionEstimator()
    re_model = CustomResNet()
    re_model.load_state_dict(torch.load("../../resnet/checkpoints/adam_best_v1.pt", map_location="cpu"))
    re.set_re_model(re_model)
    re.barCodeDecoder._halcon_handle = halcon.create_bar_code_model([], [])
    decode_results = dict()
    folder = "C:/Users/PC/Desktop/"
    result_folder = folder + "results/"
    os.makedirs(result_folder, exist_ok=True)
    files = os.listdir(folder)
    all_count = 0
    right_count = 0
    for file in files:
        # if file.endswith(".png") or file.endswith(".JPG"):
        if file.endswith("rotated_0.png"):
            all_count += 1
            res, image = re.estimate(folder + file, save_rect=False, save_cropped=True, decoder="halcon")
            # if len(res) > 0:
            #     right_count += 1
            #     data = res[0]
            #     if len(data) == 13:
            #         appendResult(decode_results, data)
            #         cv.imwrite(result_folder + data + "_" + str(decode_results[data]) + ".png",
            #                    image, [cv.IMWRITE_PNG_COMPRESSION, 0])
            # print(file, end="\t")
            # print(res)
    print("all: ", all_count)
    print("right: ", right_count)
    print("acc: ", right_count / all_count if all_count > 0 else 0)
    # print(is_valid_ean13("3051846625862"))
