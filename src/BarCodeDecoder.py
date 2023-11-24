import math
import os.path
from typing import Any

import cv2 as cv
import halcon
import onnxruntime
import pyzbar.pyzbar as pyzbar
import pyzxing
import torch
from PIL import Image
from numpy import ndarray, dtype, generic
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from decode_network.DecodeNet import DecodeNet
from src.util.constants import IMAGE_MIN_SIDE
from src.util.util import resize, is_valid_ean13
from unified_network.DetNet import DetNet


class BarCodeDecoder:
    def __init__(self, yolo_model: nn.Module = None, re_model: nn.Module = None,
                 sr_model: nn.Module = None, decode_model: nn.Module = None):
        """
        :param yolo_model: yolo目标检测模型，用于初步检测条形码位置
        :param re_model: region estimation 网络, 用于进一步确定条形码位置
        :param sr_model: super resolution 模型，用于对图片进行超分辨率重建
        """
        self.yolo_model = yolo_model
        self.re_model = re_model
        self.sr_model = sr_model
        self.decode_model = decode_model
        self.decode_session = None
        # 图片
        self._image = None
        # 图片路径，用于保存中间结果
        self._image_path = None
        # halcon条形码识别模型
        self._halcon_handle = None
        # resnet的预处理器
        self.resnet_preprocess = None
        # 区域估计偏移值
        self.re_offset = 5
        # zxing解码器
        self.zxing_reader = pyzxing.BarCodeReader()
        # 解析结果
        self.decode_result = dict()

    def set_yolo_model(self, yolo_model: nn.Module):
        self.yolo_model = yolo_model
        return self

    def set_re_model(self, re_model: nn.Module):
        self.re_model = re_model
        return self

    def set_sr_model(self, sr_model: nn.Module):
        self.sr_model = sr_model
        return self

    def set_decode_model(self, decode_model: nn.Module):
        self.decode_model = decode_model
        return self

    def detectAndDecode(self, source, decoder="halcon"):
        self._image_path = source
        if self._halcon_handle is None and decoder == "halcon":
            self._halcon_handle = halcon.create_bar_code_model([], [])
        image = cv.imread(source)
        result = self._decode(image, decoder)
        if not result:
            boxes = self.detect(source)
            if len(boxes) > 0:
                cropped = self.crop(boxes)
                result = self.decode(cropped, "halcon")
                if not result:
                    result = self.decode(cropped, "zbar")
                if not result:
                    result = self.decode(cropped, "zxing")
            else:
                print("no detection\t", source)
        return result

    def detectAndDecodeByNetwork(self, source: str) -> tuple[list[str | None], str | None]:
        self._image_path = source
        boxes = self.detect(source)
        result_type = "zbar"
        if len(boxes) == 0:
            print("no detection\t", source)
            return [], None
        cropped = self.crop(boxes, confidence=0.25)
        result = self.decode(cropped, decoder="zbar")
        if not result:
            result = self.decode(cropped, "network")
            result_type = "network"
        return result, result_type

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
            self.yolo_model = YOLO("../yolo/weights/best_v7.pt")
        detect_results = self.yolo_model.predict(source=image)
        result = []
        for r in detect_results:
            r = r.cpu().numpy()
            boxes = r.boxes.data.tolist()
            if len(boxes) > 1:
                print("multi objects: ", self._image_path)
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
        img = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imwrite(file_name, img, [cv.IMWRITE_JPEG_QUALITY, 100])

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
            if r["confidence"] >= confidence:
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
            # cv.imshow("rotated", src)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            result1 = self._decode(src, decoder)
            # if not result1:
            #     result1 = self._decode_after_optimization(src, decoder)
            result += result1
        return result

    def _decode_after_optimization(self, src, decoder):
        optimizations = [(True, True, 1), (True, True, 2), (True, False, 1), (False, True, 1)]
        for re, sr, order in optimizations:
            optimized_src = self.optim(src, re, sr, order)
            result = self._decode(optimized_src, decoder)
            if result:
                return result
        return []

    def optim(self, source: ndarray, re=True, sr=True, order=1):
        if re and sr:
            if order == 1:
                return self.up_sample(self.region_estimate(source))
            else:
                return self.region_estimate(self.up_sample(source))
        elif re:
            return self.region_estimate(source)
        elif sr:
            return self.up_sample(source)
        else:
            return source

    def region_estimate(self, source: ndarray, rect: bool = True):
        assert self.re_model is not None
        if self.resnet_preprocess is None:
            self.resnet_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        input_image = Image.fromarray(cv.cvtColor(source, cv.COLOR_BGR2RGB))
        input_image = self.resnet_preprocess(input_image)
        input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            self.re_model.eval()
            output = self.re_model(input_image)
        value1, value2, value3, value4 = output[0]
        width, height = source.shape[:2]
        x1 = max(0, int(width * value1 - 0.5 * width * value3 - self.re_offset))
        x2 = min(width, int(width * value1 + 0.5 * width * value3 + self.re_offset))
        y1 = max(0, int(height * value2 - 0.5 * height * value4 - self.re_offset))
        y2 = min(height, int(height * value2 + 0.5 * height * value4 + self.re_offset))
        if rect:
            source = cv.rectangle(source, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
            # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            ha_image = halcon.himage_from_numpy_array(image)
            gray_image = halcon.rgb1_to_gray(ha_image)
            _, content = halcon.find_bar_code(gray_image, self._halcon_handle, ["EAN-13", "Code 128"])
            result += content
        elif decoder == "zbar":
            if len(image.shape) >= 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            res = pyzbar.decode(image)
            for r in res:
                data = r.data.decode("utf-8")
                result.append(data)
        elif decoder == "zxing":
            barcode = self.zxing_reader.decode_array(image)
            if barcode:
                data = barcode[0].get("parsed")
                if data:
                    result.append(str(data))
        elif decoder == "network":
            data = self.__decode_by_network(image)
            # if is_valid_ean13(data):
            result.append(data)
        else:
            raise Exception("unsupported decoder")
        # for data in result:
        #     if len(data) == 13:
        #         self.__appendResult(data)
        #         file_name = self._image_path.split("/")[-1]
        #         save_name = self._image_path.replace(file_name, "results/" + data + "_" +
        #                                              str(self.decode_result[data]) + ".png")
        #         cv.imwrite(save_name, image, [cv.IMWRITE_PNG_COMPRESSION, 0])
        return result

    def __decode_by_network(self, image: ndarray):
        if self.decode_model is None:
            self.decode_model = DecodeNet()
            self.decode_model.load_state_dict(torch.load("../decode_network/tune/resnet50_v0.4p_adam_best.pt"))
            # self.decode_model = DetNet()
            # self.decode_model.load_state_dict(torch.load("../unified_network/tune/resnet50_v0.2_adam_best.pt"))
        if self.resnet_preprocess is None:
            self.resnet_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if self.decode_session is None:
            self.decode_session = onnxruntime.InferenceSession("../decode_network/onnx/resnet50_v0.7p_adam_best.onnx")
        input_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # input_image = Image.fromarray(image)
        # cv.imwrite("temp.png", image, [cv.IMWRITE_PNG_COMPRESSION, 0])
        # input_image = Image.open("temp.png")
        input_image = self.resnet_preprocess(input_image)
        input_image = input_image.unsqueeze(0)
        # with torch.no_grad():
        #     self.decode_model.eval()
        #     output = self.decode_model(input_image)
        output = self.decode_session.run(["output"], {'input': input_image.numpy()})[0]
        output = torch.from_numpy(output)
        # _, output = output.split([4, 130], dim=1)
        # output = output.reshape(-1, 13, 10)
        output = output.view(-1, 13, 10)
        output = nn.functional.softmax(output, dim=-1)
        _, predicted = torch.max(output, 2)
        arr = predicted.squeeze().cpu().numpy()
        result = "".join(map(str, arr))
        return result

    def __appendResult(self, data):
        if data in self.decode_result:
            self.decode_result[data] += 1
        else:
            self.decode_result[data] = 0

    def _preprocess(self, source: str) -> tuple[Any, float]:
        """
        图片预处理，保存原图和路径，然后对图片进行缩放
        :param source: 图片路径
        :return: 缩放后的图片
        """
        self._image_path = source
        self._image = cv.imread(source)
        # return resize(self._image, IMAGE_MIN_SIDE)
        return self._image, 1.0

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
        if lines is None or len(lines) == 0 or lines[0] is None:
            print("no lines detected")
            return {90: 1}, []
        deg_dists = dict()
        for line in lines[0]:
            x0, y0, x1, y1 = map(int, line[0])
            delta_y = y1 - y0
            delta_x = x1 - x0
            if delta_x == 0:
                angle = 90
            else:
                angle = round(math.degrees(math.atan2(delta_y, delta_x)))
            dist = round(math.dist((x0, y0), (x1, y1)))
            deg_dists[angle] = deg_dists.get(angle, 0) + dist
        return deg_dists, lines[0]
