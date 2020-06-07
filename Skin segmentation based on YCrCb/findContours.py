import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    sz = 7
    median = cv.medianBlur(fgmask, sz)
    contours, hierarchy = cv.findContours(median, 2, 1)

    for cnt in contours:
        hull = cv.convexHull(cnt)
        length = len(hull)
        print(length)
        # 如果凸包点集中的点个数大于5
        if length > 5:
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv.line(median, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 0, 255), 2)
    # cv.imshow('frame',fgmask)
    cv.imshow("median", median)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()