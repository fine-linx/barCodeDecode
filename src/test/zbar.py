import os

import cv2 as cv
import pyzbar.pyzbar as pyzbar


def decode(_src, binary_max=230, binary_step=2):
    if len(_src.shape) >= 3:
        _src = cv.cvtColor(_src, cv.COLOR_BGR2GRAY)
    mat = _src
    binary, _ = cv.threshold(mat, 0, 255, cv.THRESH_OTSU)
    res = []
    while (binary < binary_max) and (len(res) == 0):
        binary, mat = cv.threshold(mat, binary, 255, cv.THRESH_BINARY)
        res = pyzbar.decode(mat)
        binary += binary_step
    # res = pyzbar.decode(mat)
    if res:
        for r in res:
            print(r.data.decode("utf-8"))
        return 1
    print(res)
    return 0


if __name__ == '__main__':
    folder = "../../db/final_unresolved/rotated/cropped/"
    files = os.listdir(folder)
    for file in files:
        barcode = decode(cv.imread(folder + file))
