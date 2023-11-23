import os

import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../../yolo/weights/best_v7.pt")

    folder = "E:/work/barCode/20231117_img/detect_none/"
    files = os.listdir(folder)
    for file in files:
        result = model(folder + file)
        annotation = result[0].plot()
        annotation = cv2.resize(annotation, (1280, 720))
        cv2.imshow("detect result", annotation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
