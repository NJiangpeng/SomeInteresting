import cv2
import numpy as np
import math

##输入结果库

##摄像机输入
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ##阈值分割
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ##深复制
    drawing = np.zeros(img.shape, np.uint8)

    max_area = 0
    ##找轮廓
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)  # 0621
    ##meanshift求质心
    moments = cv2.moments(cnt)
    # print len(cnt)
    # print hull
    ##求质心公式
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

    centr = (cx, cy)
    cv2.circle(img, centr, 5, [0, 0, 255], 2)
    # cv2.circle(img,centr,5,[0,255,255],2)#0621
    # cv2.rectangle(original, p1, p2, (77, 255, 9), 1, 1)#0621

    cv2.drawContours(drawing, [cnt], 0, (255, 255, 0), 2)
    # cv2.drawContours(drawing,[hull],0,(0,0,255),2)

    cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    hull = cv2.convexHull(cnt, returnPoints=False)

    ndefects = 0  # 0622 for counting finger_number
    ###根据图像中凹凸点中的 (开始点, 结束点, 远点)的坐标, 可利用余弦定理计算两根手指之间的夹角,

    defects = cv2.convexityDefects(cnt, hull)
    if np.any(defects != None):
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            # dist = cv2.pointPolygonTest(cnt,centr,True)
            a = np.sqrt(np.square(start[0] - end[0]) + np.square(start[1] - end[1]))  # 0622
            b = np.sqrt(np.square(start[0] - far[0]) + np.square(start[1] - far[1]))  # 0622
            c = np.sqrt(np.square(end[0] - far[0]) + np.square(end[1] - far[1]))  # 0622

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # * 57#0622
            ##其必为锐角, 根据锐角的个数判别手势
            if angle <= math.pi / 2:  # 90:#0622
                ndefects = ndefects + 1  # 0622

            # cv2.line(img,start,end,[0,255,255],2)
            cv2.line(img, start, centr, [0, 255, 255], 2)
            cv2.circle(img, start, 20, [0, 255, 255], 4)
            # cv2.circle(img,end,20,[0,255,0],4)
            cv2.circle(img, far, 5, [0, 0, 255], -1)
        # print(i)

    cv2.imshow('output', drawing)
    cv2.imshow('input', img)

    k = cv2.waitKey(10)
    # Esc
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()