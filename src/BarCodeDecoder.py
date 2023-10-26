import math
from typing import Any

import cv2 as cv
import halcon
import pyzbar.pyzbar as pyzbar
import torch
from PIL import Image
from numpy import ndarray, dtype, generic
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

from util.constants import IMAGE_MIN_SIDE
from util.util import resize


class BarCodeDecoder:
    def __init__(self, yolo_model: nn.Module = None, re_model: nn.Module = None, sr_model: nn.Module = None):
        """
        :param yolo_model: yolo目标检测模型，用于初步检测条形码位置
        :param re_model: region estimation 网络, 用于进一步确定条形码位置
        :param sr_model: super resolution 模型，用于对图片进行超分辨率重建
        """
        self.yolo_model = yolo_model
        self.re_model = re_model
        self.sr_model = sr_model
        # 图片
        self._image = None
        # 图片路径，用于保存中间结果
        self._image_path = None
        # halcon条形码识别模型
        self._halcon_handle = None
        # 区域估计时的预处理器
        self.re_preprocess = None
        # 区域估计偏移值
        self.re_offset = 5

    def set_yolo_model(self, yolo_model: nn.Module):
        self.yolo_model = yolo_model
        return self

    def set_re_model(self, re_model: nn.Module):
        self.re_model = re_model
        return self

    def set_sr_model(self, sr_model: nn.Module):
        self.sr_model = sr_model
        return self

    def detectAndDecode(self, source, decoder="halcon"):
        if self._halcon_handle is None and decoder == "halcon":
            self._halcon_handle = halcon.create_bar_code_model([], [])
        image = cv.imread(source)
        result = self._decode(image, decoder)
        if not result:
            boxes = self.detect(source)
            if len(boxes) > 0:
                cropped = self.crop(boxes)
                result = self.decode(cropped, decoder)
        return result

    def detect(self, source: str, save_rect: bool = False, save_dir: str = None) -> list[dict[str, int | float]]:
        """
        使用yolo对图片中的条形码进行检测
        :param source: 图片名称
        :param save_rect: 是否保存检测结果
        :param save_dir: 保存路径
        :return: yolo检测的box结果列表，包括左上角坐标，右下角坐标，置信度，类别id
        """
        image, coeff_expansion = self._preprocess(source)
        if self.yolo_model is None:
            self.yolo_model = YOLO("../yolo/weights/best_v1.pt")
        detect_results = self.yolo_model.predict(source=image)
        result = []
        for r in detect_results:
            r = r.cpu().numpy()
            boxes = r.boxes.data.tolist()
            for box in boxes:
                result.append(
                    {
                        "x1": int(box[0] * coeff_expansion),
                        "y1": int(box[1] * coeff_expansion),
                        "x2": int(box[2] * coeff_expansion),
                        "y2": int(box[3] * coeff_expansion),
                        "confidence": box[4],
                        "class_id": box[5]
                    }
                )
        if save_rect:
            file_name = self._image_path.split("/")[-1]
            ext = "." + file_name.split(".")[-1]
            if save_dir is None:
                save_name = self._image_path.replace(ext, "_rect" + ext)
            else:
                save_name = file_name.replace(ext, "_rect" + ext)
                save_name = save_dir + "/" + save_name
            self._save_rect(self._image, result, save_name)
        return result

    @staticmethod
    def _save_rect(image: cv.Mat, boxes: list[dict[str, int | float]], file_name: str):
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            image = cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imwrite(file_name, image, [cv.IMWRITE_JPEG_QUALITY, 100])

    def crop(self, boxes: list[dict[str, int | float]], source: cv.Mat = None, save: bool = False, save_dir: str = None,
             confidence: float = 0.5):
        """
        将条形码从图片中裁剪出来
        :param boxes: 条形码所在的框
        :param source: 图片
        :param save: 是否保存
        :param save_dir: 保存的路径
        :param confidence: 条形码的置信度，高于置信度才认为是条形码
        :return: 裁剪出的条形码图片列表
        """
        if source is None:
            assert self._image is not None
            source = self._image
        result = []
        for idx, r in enumerate(boxes):
            cropped_image = source[r["y1"]:r["y2"], r["x1"]:r["x2"]]
            if r["confidence"] > confidence:
                result.append(cropped_image)
                if save:
                    file_name = self._image_path.split("/")[-1]
                    ext = "." + file_name.split(".")[-1]
                    if save_dir is None:
                        save_name = self._image_path.replace(ext, "_cropped_" + str(idx) + ".png")
                        cv.imwrite(save_name, cropped_image, [cv.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        save_name = file_name.replace(ext, "_cropped_" + str(idx) + ".png")
                        cv.imwrite(save_dir + "/" + save_name, cropped_image, [cv.IMWRITE_PNG_COMPRESSION, 0])
        return result

    def decode(self, sources: list[ndarray], decoder: str = "zbar", rotate: bool = True, save_rotated: bool = False,
               save_dir: str = None) -> list[str]:
        if self._halcon_handle is None and decoder == "halcon":
            self._halcon_handle = halcon.create_bar_code_model([], [])
        result = []
        for idx, src in enumerate(sources):
            if len(src) == 0:
                continue
            if rotate:
                src, lines = self._rotate(src)
            if save_rotated and rotate:
                file_name = self._image_path.split("/")[-1]
                ext = "." + file_name.split(".")[-1]
                if save_dir is None:
                    save_name = self._image_path.replace(ext, "_rotated_" + str(idx) + ".png")
                    cv.imwrite(save_name, src, [cv.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    save_name = file_name.replace(ext, "_rotated_" + str(idx) + ".png")
                    cv.imwrite(save_dir + "/" + save_name, src, [cv.IMWRITE_PNG_COMPRESSION, 0])
            result1 = self._decode(src, decoder)
            if not result1:
                # 进一步确定位置
                re_image = self.region_estimate(src)
                # 上采样超分
                sr_image = self.up_sample(re_image)
                result1 = self._decode(sr_image, decoder)
            result += result1
        return result

    def region_estimate(self, source: ndarray):
        assert self.re_model is not None
        if self.re_preprocess is None:
            self.re_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        input_image = Image.fromarray(cv.cvtColor(source, cv.COLOR_BGR2RGB))
        input_image = self.re_preprocess(input_image)
        input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            self.re_model.eval()
            output = self.re_model(input_image)
        value1, value2, value3, value4 = output[0]
        width, height = source.shape[:2]
        x1 = int(width * value1 - 0.5 * width * value3 - self.re_offset)
        x2 = int(width * value1 + 0.5 * width * value3 + self.re_offset)
        y1 = int(height * value2 - 0.5 * height * value4 - self.re_offset)
        y2 = int(height * value2 + 0.5 * height * value4 + self.re_offset)
        cropped_img = source[y1:y2, x1:x2]
        return cropped_img

    def up_sample(self, source: ndarray):
        assert self.sr_model is not None
        sr_img = self.sr_model.upsample(source)
        return sr_img

    def _decode(self, image: ndarray, decoder: str) -> list[str]:
        """
        :param image: 图片
        :param decoder: 解码方式
        :return: 解码结果
        """
        result = []
        if decoder == "halcon":
            image = halcon.himage_from_numpy_array(image)
            gray_image = halcon.rgb1_to_gray(image)
            _, content = halcon.find_bar_code(gray_image, self._halcon_handle, ["EAN-13", "Code 128"])
            result += content
        elif decoder == "zbar":
            if len(image.shape) >= 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            res = pyzbar.decode(image)
            for r in res:
                data = r.data.decode("utf-8")
                result.append(data)
        else:
            raise Exception("unsupported decoder")
        return result

    def _preprocess(self, source: str) -> tuple[Any, float]:
        """
        图片预处理，保存原图和路径，然后对图片进行缩放
        :param source: 图片路径
        :return: 缩放后的图片
        """
        self._image_path = source
        self._image = cv.imread(source)
        return resize(self._image, IMAGE_MIN_SIDE)

    @staticmethod
    def _rotate(image: cv.Mat) -> tuple[cv.Mat | ndarray[Any, dtype[generic]] | ndarray, Any]:
        """
        旋转图片，并返回图片中的线段
        :param image: 图片
        :return: 旋转后的图片，线段列表
        """
        deg_dists, lines = BarCodeDecoder._lsd(image)
        rotate_angle = max(deg_dists, key=deg_dists.get) + 90
        # 将图片填充到原始图片的外接圆的外接正方形大小，以保证无信息丢失
        height, width = image.shape[:2]
        side_length = round(math.sqrt(height ** 2 + width ** 2))
        left = (side_length - width + 1) // 2
        top = (side_length - height + 1) // 2
        src = cv.copyMakeBorder(image, top, top, left, left, cv.BORDER_CONSTANT, value=(0, 0, 0))
        center = (side_length // 2, side_length // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, rotate_angle, 1.0)
        rotation_image = cv.warpAffine(src, rotation_matrix, (side_length, side_length))
        return rotation_image, lines

    @staticmethod
    def _lsd(image: cv.Mat) -> tuple[dict[float | int, int], cv.Mat | ndarray[Any, dtype[generic]] | ndarray]:
        """
        使用lsd算法对图片进行检测，以获得旋转角度
        :param image: 图片
        :return: 角度-长度值，线段列表
        """
        if len(image.shape) >= 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        lsd = cv.createLineSegmentDetector(0, scale=1)
        lines = lsd.detect(image)
        deg_dists = dict()
        for line in lines[0]:
            x0 = int(round(line[0][0]))
            y0 = int(round(line[0][1]))
            x1 = int(round(line[0][2]))
            y1 = int(round(line[0][3]))
            delta_y = y1 - y0
            delta_x = x1 - x0
            if delta_x == 0:
                angle = 90.0
            else:
                angle = round(math.degrees(math.atan2(delta_y, delta_x)))
            dist = round(math.dist((x0, y0), (x1, y1)))
            if angle in deg_dists:
                deg_dists[angle] += dist
            else:
                deg_dists[angle] = dist
        return deg_dists, lines[0]
