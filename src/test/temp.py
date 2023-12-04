import cv2
import numpy as np

# 1. 加载图像
image = cv2.imread('../../db/20231019/folder_1/rotated/20231019073605330048_S_02_rotated_0.png')

# 图像尺寸
height, width, _ = image.shape

# 设置反光区域
reflection_intensity = 0.7  # 反光强度
reflection_size = 50  # 反光区域的大小

# 生成反光噪声
reflection = np.zeros_like(image, dtype=np.uint8)
cv2.rectangle(reflection, (width // 2, 0), (width // 2 + reflection_size, height), (255, 255, 255), -1)
blurred_reflection = cv2.GaussianBlur(reflection, (25, 25), 0)  # 模糊反光区域
blurred_reflection = blurred_reflection.astype(np.uint8) * reflection_intensity  # 转换类型并根据强度调整反光的亮度

# 将反光添加到图像上
blurred_reflection = blurred_reflection.astype(image.dtype)  # 调整数据类型与原图像相同
noisy_image = cv2.addWeighted(image, 1, blurred_reflection, 1, 0)

# 显示原始图像和带反光噪声的图像
cv2.imshow('Original Image', image)
cv2.imshow('Image with Reflection Noise', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
