import os

import cv2 as cv

from src.BarCodeDecoder import BarCodeDecoder


class UpSample:
    def __init__(self, _sr_model=None):
        self.sr_model = _sr_model
        self.barCodeDecoder = BarCodeDecoder()

    def set_sr_model(self, _sr_model):
        self.sr_model = _sr_model

    def up_sample(self, source, save=False, _decoder="zbar"):
        assert self.sr_model is not None
        img = cv.imread(source)
        img = self.sr_model.upsample(img)
        result = self.barCodeDecoder.decode([img], decoder=_decoder, rotate=False)
        if save:
            file_name = source.split("/")[-1]
            save_dir = source.replace(file_name, "up_sample/")
            os.makedirs(save_dir, exist_ok=True)
            cv.imwrite(save_dir + file_name, img, [cv.IMWRITE_PNG_COMPRESSION, 0])
        return result, img


if __name__ == '__main__':
    # 超分模型
    model_path = "../../sr_models/ESPCN/ESPCN_x2.pb"
    sr = cv.dnn_superres.DnnSuperResImpl.create()
    sr.readModel(model_path)
    sr.setModel("espcn", 2)
    decoder = UpSample()
    decoder.set_sr_model(sr)

    folder = "../../db/barCodeDB2/rotated/unresolved/halcon/cropped/"
    files = os.listdir(folder)
    all_count = 0
    right_count = 0
    for file in files:
        if file.endswith(".png"):
            all_count += 1
            res, _ = decoder.up_sample(folder + file, save=True, _decoder="halcon")
            if len(res) > 0:
                right_count += 1
            print(file, end="\t")
            print(res)
    print("all: ", all_count)
    print("right: ", right_count)
    print("acc: ", right_count / all_count if all_count > 0 else 0)
