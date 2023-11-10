import os
import shutil

import cv2 as cv
import halcon as ha
from pyzbar import pyzbar


class DetectAndDecode:
    def __init__(self):
        self.bar_det = cv.barcode.BarcodeDetector()
        self.barcode_count = 0
        self.right_count = 0
        self.unresolved_path = {"halcon": "../unresolved/halcon/",
                                "opencv": "../unresolved/opencv/"
                                }

    def detectAndDecode(self, path: str, label: str, isHalcon=False):
        self.barcode_count += 1
        if label[0] == '0':
            label = label[1:]
        if isHalcon:
            image = ha.read_image(path)
            grayImage = ha.rgb1_to_gray(image)
            # width, height = ha.get_image_size(grayImage)
            # grayImage = ha.emphasize(grayImage, width[0], height[0], 1)
            barCodeHandle = ha.create_bar_code_model([], [])
            symbolRegions, content = ha.find_bar_code(grayImage, barCodeHandle, ['EAN-13', 'Code 128', 'auto'])
            if label in content:
                # if content:
                self.right_count += 1
                print(path, end="\t")
                print(content)
            else:
                print(path)
                file_name = path.split("/")[-1]
                unresolved_path = path.replace(file_name, "unresolved/")
                os.makedirs(unresolved_path, exist_ok=True)
                shutil.copy(path, unresolved_path + file_name)
                # shutil.copy(path, self.unresolved_path["halcon"] + file_name)
        else:
            src = cv.imread(path)
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            res = pyzbar.decode(src)
            if res and label == res[0].data.decode("utf-8"):
                self.right_count += 1
                print(path, end="\t")
                print(res[0].data.decode("utf-8"))
            else:
                print(path)
                file_name = path.split("/")[-1]
                # shutil.copy(path, self.unresolved_path["opencv"] + file_name)
                unresolved_path = path.replace(file_name, "unresolved/zbar/")
                os.makedirs(unresolved_path, exist_ok=True)
                shutil.copy(path, unresolved_path + file_name)

    @staticmethod
    def detect(path: str):
        image = ha.read_image(path)
        grayImage = ha.rgb1_to_gray(image)
        barCodeHandle = ha.create_bar_code_model([], [])
        symbolRegions, content = ha.find_bar_code(grayImage, barCodeHandle, ['EAN-13', 'Code 128'])
        if not content:
            file_name = path.split("/")[-1]
            shutil.copy(path, "../maybe_none/" + file_name)

    def getAccuracy(self) -> float:
        print("all: ", self.barcode_count)
        print("right: ", self.right_count)
        return self.right_count / self.barcode_count if self.barcode_count > 0 else 0
