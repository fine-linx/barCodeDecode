import os
import shutil

import cv2
import numpy as np
from skimage import util as sk_util

if __name__ == '__main__':
    # 读取图片
    data_type = "valid/"
    folder = "E:/work/barCode/net_dataset3/" + data_type
    target_folder = "E:/work/barCode/net_dataset4/" + data_type
    os.makedirs(target_folder, exist_ok=True)
    files = os.listdir(folder)
    kernel = np.ones((3, 3), np.uint8)
    for file in files:
        if file.endswith(".png"):
            shutil.copy(folder + file, target_folder + file)
            image = cv2.imread(folder + file)
            for noise_type in ['localvar']:
                noisy_image = sk_util.random_noise(image, mode=noise_type)
                noisy_image = (noisy_image * 255).astype(np.uint8)
                file_name = file.split('.')[0] + '_' + noise_type + '.png'
                cv2.imwrite(target_folder + file_name, noisy_image)
            noisy_image = cv2.GaussianBlur(image, (5, 5), 0)
            file_name = file.split('.')[0] + '_' + 'gaussian' + '.png'
            cv2.imwrite(target_folder + file_name, noisy_image)
            noisy_image = cv2.dilate(image, kernel, iterations=1)
            file_name = file.split('.')[0] + '_' + 'dilate' + '.png'
            cv2.imwrite(target_folder + file_name, noisy_image)
            noisy_image = cv2.erode(image, kernel, iterations=1)
            file_name = file.split('.')[0] + '_' + 'erode' + '.png'
            cv2.imwrite(target_folder + file_name, noisy_image)
