import os
import random
import shutil
from typing import Any

import cv2 as cv
from cv2 import Mat
from numpy import ndarray, dtype, generic


def resize(src: cv.Mat, _min_side) -> tuple[Mat | ndarray | ndarray[Any, dtype[generic | generic]], float]:
    """
    :param src: 原图
    :param _min_side: 缩放后图片的最小边
    :return: 缩放后的图，缩放比例
    """
    height, width = src.shape[0], src.shape[1]
    min_side = min(height, width)
    coeff_expansion = 1.0
    if min_side > _min_side:
        coeff_expansion = min_side / _min_side
        width = round(src.shape[1] / coeff_expansion)
        height = round(src.shape[0] / coeff_expansion)
        src = cv.resize(src, (width, height))
    return src, coeff_expansion


def randomSelect(folder_path: str, copy_path: str, ratio: float = 0.2) -> None:
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if random.random() <= ratio:
            shutil.copy(folder_path + file_name, copy_path + file_name)


def is_valid_ean13(barcode):
    # 确保输入是一个13位的数字字符串
    if not barcode.isdigit() or len(barcode) != 13:
        return False

    # 计算校验位
    odd_sum = sum(int(barcode[i]) for i in range(0, 12, 2))
    even_sum = sum(int(barcode[i]) for i in range(1, 12, 2))
    total = odd_sum + even_sum * 3
    checksum = (10 - (total % 10)) % 10

    # 检查校验位是否与计算的校验位相符
    return int(barcode[12]) == checksum
