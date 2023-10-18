import math

import cv2 as cv


def rotation(src, angle):
    height, width = src.shape[:2]
    side_length = round(math.sqrt(height ** 2 + width ** 2))
    left = (side_length - width + 1) // 2
    top = (side_length - height + 1) // 2
    src = cv.copyMakeBorder(src, top, top, left, left, cv.BORDER_CONSTANT, value=(0, 0, 0))
    cv.imshow("bordered", src)
    height, width = src.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotation_image = cv.warpAffine(src, rotation_matrix, (width, height))
    return rotation_image


if __name__ == '__main__':
    img = cv.imread("D:/Linx_work/yolo_barcode/src/cropped_images/20230206091306472969_S_01_cropped_0.JPG")
    img = rotation(img, 160)
    cv.imshow("rotated", img)
    # cv.imwrite("../barCodeDB2/cropped_images/20230206091309385972_S_01_cropped_0.jpg", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
