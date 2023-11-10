import math
import statistics

import cv2 as cv
import matplotlib.pyplot as plt


def lsd_detection(src: cv.UMat):
    lsd = cv.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(src)
    degrees = dict()
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
            angle_degrees = 90
        else:
            angle_radians = math.atan2(delta_y, delta_x)
            angle_degrees = round(math.degrees(angle_radians))
        angle_degrees = (angle_degrees + 180) % 180
        dist = round(math.dist((x0, y0), (x1, y1)))
        if angle_degrees in degrees:
            degrees[angle_degrees] += dist
        else:
            degrees[angle_degrees] = dist
    return src, degrees


if __name__ == '__main__':
    img = cv.imread("C:/Users/PC/Desktop/8033873185026_2_cropped_0.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img, degrees = lsd_detection(img)
    print(degrees)
    print(statistics.mode(degrees))
    plt.figure(1)
    x = list(degrees.keys())
    y = list(degrees.values())
    # plt.hist(degrees, bins=180, edgecolor='k')
    plt.bar(x, y)
    plt.xlabel("angle")
    plt.ylabel("length")
    plt.xticks()
    plt.title("angle histogram")
    plt.show()
    cv.imshow("line", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
