import os
import shutil

import cv2 as cv
import pyzbar.pyzbar as pyzbar
import pyzxing

reader = pyzxing.BarCodeReader()


def zxing_decode(file_path):
    results = reader.decode(file_path)
    if results:
        code_text = results[0].get("parsed")
        print(code_text)
        return 1
    return 0


# 超分模型
model_path = "../../yolo_barcode/models/ESPCN/ESPCN_x2.pb"
sr = cv.dnn_superres.DnnSuperResImpl.create()
sr.readModel(model_path)
sr.setModel("espcn", 2)


def decode(_src, binary_max=230, binary_step=2):
    if len(_src.shape) >= 3:
        _src = cv.cvtColor(_src, cv.COLOR_BGR2GRAY)
    mat = _src
    res = pyzbar.decode(mat)
    if res:
        for r in res:
            print(r.data.decode("utf-8"))
        return 1
    # else:
    #     mat = sr.upsample(mat)
    #     # cv.imshow("upsample", mat)
    #     res = pyzbar.decode(mat)
    #     if res:
    #         print(res)
    #         return 1
    print(res)
    return 0


bar_det = cv.barcode.BarcodeDetector()


def decodeCV(_src):
    content, _, _ = bar_det.detectAndDecode(_src)
    print(content)
    if content:
        return 1
    return 0


if __name__ == '__main__':
    # folder = "D:/Linx_work/yolo_barcode/src/rotated_images/db2/"
    folder = "../db/barCodeDB2/rotated/"
    files = os.listdir(folder)
    all_barcode = 0
    right = 0
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".JPG"):
            all_barcode += 1
            src = cv.imread(folder + file)
            # src = sr.upsample(src)
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            cnt = decode(src)
            if cnt == 0:
                shutil.copy(folder + file, folder + "unresolved/zbar/" + file)
            # cnt = zxing_decode(folder + file)
            right += cnt
            # right += zxing_decode(src)
    print("all: ", len(files))
    print("right: ", right)
    print("acc: ", right / all_barcode if all_barcode > 0 else 0)
