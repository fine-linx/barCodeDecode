import os

import cv2 as cv

if __name__ == '__main__':
    decoder = cv.barcode.BarcodeDetector()
    folder = "../../db/final_unresolved/rotated/cropped/"
    files = os.listdir(folder)
    for file in files:
        img = cv.imread(folder + file)
        code, _, _ = decoder.detectAndDecode(img)
        print("code: ", code)
