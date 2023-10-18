import os
import shutil
import time

import cv2 as cv
import torch
from ultralytics import YOLO

from BarCodeDecoder import BarCodeDecoder
from detectAndDecode import DetectAndDecode
from resnet.CustomResNet import CustomResNet


def detectAll(isHalcon=False):
    decoder = DetectAndDecode()
    folder_path = "../db/barCodeDB2/"
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if file_name.endswith(".JPG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
            label = file_name.split("_")[0]
            file_path = folder_path + file_name
            decoder.detectAndDecode(file_path, label, isHalcon=isHalcon)
    print("Accuracy: ", decoder.getAccuracy())


def main():
    # yolo模型
    yolo_model = YOLO("../yolo/weights/best_v1.pt")
    # 区域估计模型
    re_model = CustomResNet()
    re_model.load_state_dict(torch.load("../resnet/checkpoints/adam_best_v1.pt", map_location="cpu"))
    # 超分模型
    sr_model_path = "../sr_models/ESPCN/ESPCN_x2.pb"
    sr_model = cv.dnn_superres.DnnSuperResImpl.create()
    sr_model.readModel(sr_model_path)
    sr_model.setModel("espcn", 2)

    decoder = BarCodeDecoder()
    decoder.set_yolo_model(yolo_model).set_sr_model(sr_model).set_re_model(re_model)

    folder = "../db/barCodeDB2/"
    detect_none_path = folder + "detect_none/"
    cropped_path = folder + "cropped/"
    rotated_path = folder + "rotated/"
    unresolved_path = folder + "unresolved/halcon/"
    rect_path = folder + "rect/"
    os.makedirs(detect_none_path, exist_ok=True)
    os.makedirs(cropped_path, exist_ok=True)
    os.makedirs(rotated_path, exist_ok=True)
    os.makedirs(unresolved_path, exist_ok=True)
    os.makedirs(rect_path, exist_ok=True)

    files = os.listdir(folder)
    all_count = 0
    right_count = 0
    for file in files:
        if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png"):
            all_count += 1
            file_path = folder + file
            # boxes = decoder.detect(file_path, save_rect=False, save_dir=rect_path)
            # if len(boxes) == 0:
            #     # 没有检测到
            #     shutil.copy(file_path, detect_none_path + file)
            #     result = decoder.decode([cv.imread(file_path)], decoder="zbar", rotate=False)
            # else:
            #     cropped = decoder.crop(boxes, save=False, save_dir=cropped_path)
            #     result = decoder.decode(cropped, decoder="zbar", save_rotated=False, save_dir=rotated_path)
            result = decoder.detectAndDecode(file_path)
            if len(result) > 0:
                right_count += 1
            else:
                shutil.copy(file_path, unresolved_path + file)
            print(file_path, end="\t")
            print(result)
    print("all: ", all_count)
    print("right: ", right_count)
    print("acc: ", right_count / all_count if all_count > 0 else 0)


if __name__ == '__main__':
    t1 = time.time()
    main()
    # detectAll(True)
    t2 = time.time()
    print("total time: %s ms" % ((t2 - t1) * 1000))
