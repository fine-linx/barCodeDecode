import os
import shutil
import time

import cv2 as cv
import torch
from torch import nn
from torchvision import models
from ultralytics import YOLO

from BarCodeDecoder import BarCodeDecoder
from decode_network.DecodeNet import DecodeNet, DecodeNetBinary
from detectAndDecode import DetectAndDecode
from resnet.CustomResNet import CustomResNet


# 关闭yolo日志
# import os
#
# os.environ["YOLO_VERBOSE"] = str(False)


def detectAll(isHalcon=False):
    decoder = DetectAndDecode()
    folder_path = "E:/work/barCode/net_dataset3/test/"
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if file_name.endswith(".JPG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
            label = file_name.split("_")[0]
            file_path = folder_path + file_name
            decoder.detectAndDecode(file_path, label, isHalcon=isHalcon)
    print("Accuracy: ", decoder.getAccuracy())


def main():
    # yolo模型
    # yolo_model = YOLO("../yolo/weights/best_s_v1.pt")
    yolo_model = YOLO("../yolo/weights/best_v7.pt")
    # 区域估计模型
    # re_model = CustomResNet()
    # re_model.load_state_dict(torch.load("../resnet/checkpoints/adam_best_v1.pt"))
    # 超分模型
    # sr_model_path = "../sr_models/ESPCN/ESPCN_x2.pb"
    # sr_model = cv.dnn_superres.DnnSuperResImpl.create()
    # sr_model.readModel(sr_model_path)
    # sr_model.setModel("espcn", 2)
    # 条形码识别网络
    # decode_network = DecodeNetBinary()
    # decode_network.load_state_dict(torch.load("../decode_network/predict_binary/resnet50_v0.5p_adam_best.pt"))
    isBinary = True
    if isBinary:
        decode_network = DecodeNetBinary()
        decode_network.load_state_dict(
            torch.load("../decode_network/predict_binary/resnet50_cropped_v0.2p_adam_best.pt"))
        # decode_network = models.efficientnet_b0()
        # decode_network.classifier = nn.Sequential(
        #     nn.Dropout(0.2, inplace=True),
        #     nn.Linear(1280, 84, bias=True),
        #     nn.Sigmoid()
        # )
        # decode_network.load_state_dict(
        #     torch.load("../decode_network/predict_binary/efficientNet_cropped_v0.1p_adam_best.pt"))
    else:
        decode_network = DecodeNet()
        decode_network.load_state_dict(torch.load("../decode_network/tune/resnet50_v0.8p_adam_best.pt"))
    # decode_network.eval()
    decoder = BarCodeDecoder().set_decode_model(decode_network).set_yolo_model(yolo_model)
    # decoder = BarCodeDecoder().set_yolo_model(yolo_model)
    # decoder.set_yolo_model(yolo_model).set_sr_model(sr_model).set_re_model(re_model)
    decode_method = "network_binary"
    folder = "E:/work/barCode/20231130_img/folder_1/"
    detect_none_path = folder + "detect_none/"
    cropped_path = folder + "cropped/"
    rotated_path = folder + "rotated/"
    model_type = "pytorch"
    unresolved_path = folder + "unresolved/" + decode_method + "/" + model_type + "/"
    rect_path = folder + "rect/"
    os.makedirs(detect_none_path, exist_ok=True)
    os.makedirs(cropped_path, exist_ok=True)
    os.makedirs(rotated_path, exist_ok=True)
    os.makedirs(unresolved_path, exist_ok=True)
    os.makedirs(rect_path, exist_ok=True)
    files = os.listdir(folder)
    all_count = 0
    right_count = 0
    zbar_count, network_count = 0, 0
    t1 = time.time()
    # file_set = get_rect()
    for idx, file in enumerate(files):
        # if idx >= 100:
        #     break
        if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png") or file.endswith("BMP"):
            all_count += 1
            file_path = folder + file
            label = file.split("_")[0]
            # boxes = decoder.detect(file_path, save_rect=False, save_dir=rect_path)
            # if len(boxes) == 0:
            #     # 没有检测到
            #     shutil.copy(file_path, detect_none_path + file)
            #     # result = decoder.decode([cv.imread(file_path)], decoder=decode_method, rotate=True)
            # else:
            #     cropped = decoder.crop(boxes, save=True, save_dir=cropped_path, confidence=0.5)
            #     result = decoder.decode(cropped, decoder=decode_method, save_rotated=True, save_dir=rotated_path)
            # result = decoder.detectAndDecode(file_path, decoder=decode_method)
            result, result_type = decoder.detectAndDecodeByNetwork(file_path, decoder=decode_method)
            if result_type is None:
                shutil.copy(file_path, detect_none_path + file)
            if label in result:
                print("right", end="\t")
                right_count += 1
                if result_type == "zbar":
                    zbar_count += 1
                else:
                    network_count += 1
            else:
                print("wrong", end="\t")
                shutil.copy(file_path, unresolved_path + file)
            # # if len(result) > 0:
            #     right_count += 1
            # else:
            #     shutil.copy(file_path, unresolved_path + file)
            print(all_count, end="\t")
            print(file_path, end="\t")
            print(result)
    print("all: ", all_count)
    print(f"right: {right_count}, zbar: {zbar_count}, network: {network_count}")
    print("acc: ", right_count / all_count if all_count > 0 else 0)
    t2 = time.time()
    print("total time: %.4f ms, per image: %.4f ms" % ((t2 - t1) * 1000, (t2 - t1) * 1000 / all_count))


def get_rect():
    folder = "E:/work/barCode/20231119_img/"
    rect_path = os.path.join(folder, "rect")
    file_set = {file.replace("_rect", "") for file in os.listdir(rect_path)}
    return file_set


if __name__ == '__main__':
    # t1 = time.time()
    main()
    # detectAll(True)
    # t2 = time.time()
    # print("total time: %s ms" % ((t2 - t1) * 1000))
