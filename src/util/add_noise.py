import os

import cv2
import numpy as np
from skimage import util as sk_util

if __name__ == '__main__':
    # 读取图片
    folder = "E:/work/barCode/yolo_dataset/images/"
    target_folder = folder
    os.makedirs(target_folder, exist_ok=True)
    files = os.listdir(folder)
    kernel = np.ones((3, 3), np.uint8)
    for file in files:
        if file.endswith(".jpg"):
            image = cv2.imread(folder + file)
            for noise_type in ['localvar']:
                noisy_image = sk_util.random_noise(image, mode=noise_type)
                noisy_image = (noisy_image * 255).astype(np.uint8)
                file_name = file.split('.')[0] + '_' + noise_type + '.jpg'
                cv2.imwrite(target_folder + file_name, noisy_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            noisy_image = cv2.GaussianBlur(image, (5, 5), 0)
            file_name = file.split('.')[0] + '_' + 'gaussian' + '.jpg'
            cv2.imwrite(target_folder + file_name, noisy_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            noisy_image = cv2.dilate(image, kernel, iterations=1)
            file_name = file.split('.')[0] + '_' + 'dilate' + '.jpg'
            cv2.imwrite(target_folder + file_name, noisy_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            noisy_image = cv2.erode(image, kernel, iterations=1)
            file_name = file.split('.')[0] + '_' + 'erode' + '.jpg'
            cv2.imwrite(target_folder + file_name, noisy_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
