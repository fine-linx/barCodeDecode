import math
import statistics

import cv2 as cv
import matplotlib.pyplot as plt


def lsd_detection(src: cv.UMat):
    lsd = cv.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(src)
    degrees = []
    for dline in dlines[0]:
        # print(dline)
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv.line(src, (x0, y0), (x1, y1), 255, 1, cv.LINE_AA)
        delta_y = y1 - y0
        delta_x = x1 - x0
        if delta_x == 0:
            angle_degrees = 90.0
        else:
            angle_radians = math.atan2(delta_y, delta_x)
            angle_degrees = math.degrees(angle_radians)
        degrees.append(int(angle_degrees))
    return src, degrees


if __name__ == '__main__':
    img = cv.imread("D:/Linx_work/yolo_barcode/src/unresolved/rotated/20230206091309385972_S_01_rotated_0.JPG")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img, degrees = lsd_detection(img)
    print(degrees)
    print(statistics.mode(degrees))
    plt.figure(1)
    plt.hist(degrees, bins=180, edgecolor='k')
    plt.xlabel("angle")
    plt.ylabel("length")
    plt.title("angle histogram")
    plt.show()
    cv.imshow("line", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
