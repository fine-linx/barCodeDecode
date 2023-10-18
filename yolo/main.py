import os
from typing import Any

import cv2 as cv
from cv2 import Mat
from numpy import ndarray, dtype, generic
from ultralytics import YOLO

MIN_SIDE = 1024


def resize(src: cv.Mat) -> tuple[ndarray | ndarray[Any, dtype[generic | generic]] | Mat | Any, float]:
    height, width = src.shape[0], src.shape[1]
    min_side = min(height, width)
    coeff_expansion = 1
    if min_side > MIN_SIDE:
        coeff_expansion = min_side / MIN_SIDE
        width = round(src.shape[1] / coeff_expansion)
        height = round(src.shape[0] / coeff_expansion)
        src = cv.resize(src, (width, height))
    return src, coeff_expansion


# model = YOLO("yolov8n.pt")
model = YOLO("D:/Linx_work/yolo_barcode/src/weights/best_v1.pt")

# results = model.train(data="datasets/barCode.yaml", epochs=100)

# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered

# print("ok")

folder_path = "D:/Linx_work/barCode/data/"

file_names = os.listdir(folder_path)
# f = open("txt/result.txt", "a+")
for file_name in file_names:
    if file_name.endswith(".jpg") or file_name.endswith(".JPG"):
        src = cv.imread(folder_path + file_name)
        img, coeff = resize(src)

        detects = model.predict(source=img)
        result = []
        for detect in detects:
            temp = detect.cpu().numpy()
            boxes = temp.boxes.data.tolist()
            for box in boxes:
                top_left = (int(box[0] * coeff), int(box[1] * coeff))
                bottom_right = (int(box[2] * coeff), int(box[3] * coeff))
                color = (0, 255, 0)
                thickness = 2
                cv.rectangle(src, top_left, bottom_right, color, thickness)
                # result.append(
                #     {
                #         "x1": int(box[0] * coeff),
                #         "y1": int(box[1] * coeff),
                #         "x2": int(box[2] * coeff),
                #         "y2": int(box[3] * coeff),
                #         "confidence": box[4],
                #         "class_id": box[5]
                #     }
                # )
        # cv.imshow("img", src)
        cv.imwrite(folder_path + "rect/" + file_name, src)
        # print(result)
        # f.write(str(result) + "\n")
        # for idx, r in enumerate(result):
        #     cropped_img = src[r["y1"]:r["y2"], r["x1"]:r["x2"]]
        #     if r["confidence"] > 0.5:
        #         cv.imwrite("D:/Linx_work/barCode/barCodeDB2/cropped_images/" + file_name.split('.')[0] +
        #                    "_cropped_" + str(idx) + ".jpg", cropped_img)
# f.close()
