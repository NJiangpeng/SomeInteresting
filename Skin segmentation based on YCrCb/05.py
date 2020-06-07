
import cv2
import os
import numpy as np
import time
import math
from tkinter import *
from PIL import  Image as IMG
from PIL import  ImageTk



"""
加载图像（opencv，截图保存saveROI）
肤色检测（YCrCb颜色空间的Cr分量+Otsu法阈值分割算法）
图像去噪（numpy二值化处理）
轮廓提取（canny检测，cv2.findContours->傅里叶描述子Laplacian）
二次去噪（numpy二值化处理）
绘制轮廓（cv2.drawContours）
"""
class Extract_With_Fourier_Descriptors():

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.size = 0.5
        self.fx = 10
        self.fy = 355
        self.fh = 18
        # ROI起始位置
        self.x0 = 0
        self.y0 = 0
        # 画面大小
        self.width = 640
        self.height = 480
        #每个手势录制的大小
        self.numofsamples = 300
        self.counter = 0 # 计数器，记录已经录制多少图片了
        # 存储地址和初始文件夹名称
        self.gesturename = ""

        self.path = ""

        # 标识符 bool类型用来表示某些需要不断变化的状态
        self.binaryMode = False
        self.saveImg = False

        self.MIN_DESCRIPTOR = 32  # surprisingly enough, 2 descriptors are already enough

        self.center = ()
        #
        #
        self.approx = []
        # self.cnt = []
        #
        self.blue = (255, 0 , 0)
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.purple = (160, 32, 240)
        self.yellow = (0, 255, 255)
        self.indianRed = (106, 106, 255)

        self.stop = False
        self.color = False
        self.bin = False

    def find_contours(self, Laplacian):
        # binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
        h = cv2.findContours(Laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
        contour = h[0]
        contour = sorted(contour, key=cv2.contourArea, reverse=True)  # 对一系列轮廓点坐标按它们围成的区域面积进行排序
        return contour

    # 截短傅里叶描述子
    def truncate_descriptor(self, fourier_result):
        descriptors_in_use = np.fft.fftshift(fourier_result)

        # 取中间的MIN_DESCRIPTOR项描述子
        center_index = int(len(descriptors_in_use) / 2)
        low, high = center_index - int(self.MIN_DESCRIPTOR / 2), center_index + int(self.MIN_DESCRIPTOR / 2)
        descriptors_in_use = descriptors_in_use[low:high]

        descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
        return descriptors_in_use

    ##由傅里叶描述子重建轮廓图
    def reconstruct(self, img, descirptor_in_use):
        contour_reconstruct = np.fft.ifft(descirptor_in_use)
        contour_reconstruct = np.array([contour_reconstruct.real, contour_reconstruct.imag])
        contour_reconstruct = np.transpose(contour_reconstruct)
        contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
        if contour_reconstruct.min() < 0:
            contour_reconstruct -= contour_reconstruct.min()
        contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
        contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)

        black_np = np.ones(img.shape, np.uint8)  # 创建黑色幕布
        black = cv2.drawContours(black_np, contour_reconstruct, -1, (255, 255, 255), 1)  # 绘制白色轮廓
        # cv2.imshow("contour_reconstruct", black)
        # cv2.imwrite('recover.png',black)
        return black

    # 计算傅里叶描述算子,以获取轮廓
    def fourierDesciptor(self, res):
        # Laplacian算子进行八邻域检测
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        Laplacian = cv2.convertScaleAbs(dst)
        contour = self.find_contours(Laplacian) # 提取轮廓点坐标
        contour_array = contour[0][:, 0, :]  # 注意这里只保留区域面积最大的轮廓点坐标

        # 计算分离出来物体的极点: 极值点表示对象的最顶部，最底部，最右侧和最左侧的点
        cnt = contour[0]
        self.lt = tuple(cnt[cnt[:, :, 0].argmin()][0])
        self.rt = tuple(cnt[cnt[:, :, 0].argmax()][0])
        self.tt = tuple(cnt[cnt[:, :, 1].argmin()][0])
        self.bt = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # 获取质心
        M = cv2.moments(contour_array)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.center = (cx, cy)
        self.cnt = cnt

        # 创建黑色幕布
        ret_np = np.ones(dst.shape, np.uint8)
        # ret = cv2.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # 绘制白色轮廓
        ret = cv2.drawContours(ret_np, contour, -1, (255, 255, 255), 1)  # 绘制白色轮廓
        self.approx = contour[0]
        # print(contour[0])
        # cv2.imshow("ret", ret)
        Laplacian1 = cv2.convertScaleAbs(ret)
        contour1 = self.find_contours(Laplacian1)

        contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
        contours_complex.real = contour_array[:, 0]  # 横坐标作为实数部分
        contours_complex.imag = contour_array[:, 1]  # 纵坐标作为虚数部分
        fourier_result = np.fft.fft(contours_complex)  # 进行傅里叶变换
        # fourier_result = np.fft.fftshift(fourier_result)
        descirptor_in_use = self.truncate_descriptor(fourier_result)
        # 绘图显示

        black = self.reconstruct(ret, descirptor_in_use)
        cv2.imshow("black", black)
        self.contounr = contour_array
        self.binaryimage = ret
        self.black = black
        return descirptor_in_use

    def drawCountours(self,skin ,frame):
        img = skin.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        Laplacian = cv2.convertScaleAbs(dst)
        contours = self.find_contours(Laplacian)  # 提取轮廓点坐标
        # 创建黑色幕布
        ret_np = np.ones(dst.shape, np.uint8)
        # ret = cv2.drawContours(ret_np,  contours, -1, (255, 255, 255), 1)  # 绘制白色轮廓
        # frame = cv2.drawContours(frame, contours, -1, self.blue, 1)
        img= cv2.drawContours(img, contours, -1, self.blue, 3)
        cv2.imshow("img", img)

    # 保存ROI图像
    def saveROI(self, img):
        global path, counter, gesturename, saveImg
        if self.counter > self.numofsamples:
            # 恢复到初始值，以便后面继续录制手势
            self.saveImg = False
            self.gesturename = ''
            self.counter = 0
            return

        self.counter += 1
        name = self.gesturename + str(self.counter)  # 给录制的手势命名
        if self.color == True:
            name = name +"_color"
        else:
            name = name + "_bin"
        print("Saving img: ", name)
        cv2.imwrite(self.path + name + '.png', img)  # 写入文件
        time.sleep(0.05)

    ####YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
    def skinMask(self, roi):
        YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
        (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
        res = cv2.bitwise_and(roi, roi, mask=skin)
        return res

    def binaryMask(self, frame):
        #我们这里,整个图片都是我们的ROI,因此不再重新设置ROI
        roi = frame
        # cv2.imshow("roi", roi)
        skin = self.skinMask(roi) #通过肤色检测,获取只含肤色的部分
        # cv2.imshow("skin", skin)

        # 获取傅里叶描述算子的轮廓点
        fourier_desciptor= self.fourierDesciptor(skin)  # 获取傅里叶描述算子的轮廓点

        if self.saveImg == True and self.color == True:
            self.saveROI(skin)
        elif self.saveImg == True and self.bin == True:
            self.saveROI(self.binaryimage)
        return skin

    def trackHand(self):
        cnt = cv2.approxPolyDP(self.cnt, 0.01 * cv2.arcLength(self.cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)

        ndefects = 0  # 0622 for counting finger_number
        # 根据图像中凹凸点中的 (开始点, 结束点, 远点)的坐标, 可利用余弦定理计算两根手指之间的夹角,
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
                cv2.line(self.img, start, self.center, self.blue, 2)
                cv2.circle(self.img, start, 12, self.green, 4)
                # cv2.circle(img,end,20,[0,255,0],4)
                cv2.circle(self.img, far, 5, self.red, -1)

        cv2.imshow("img", self.img)

    def findFinger(self):
        print(0)

    def pointPose(self, frame, skin):
        # 在原来的fram上显示 4个极点的位置
        cv2.circle(frame, self.lt, 5, (106, 106, 255), -1)
        cv2.circle(frame, self.rt, 7, (0, 255, 255), -1)
        cv2.circle(frame, self.tt, 7, (255, 0, 255), -1)
        cv2.circle(frame, self.bt, 7, (238, 103, 122), -1)
        # 连接极点 与 质心
        # cv2.line(frame, self.center, self.lt, (106, 106, 255), 2)
        # cv2.line(frame, self.center, self.rt, (0, 255, 255), 2)
        # cv2.line(frame, self.center, self.tt, (255, 0, 255), 2)
        # cv2.line(frame, self.center, self.bt, (238, 103, 122), 2)

        cv2.circle(skin, self.lt, 5, (106, 106, 255), -1)
        cv2.circle(skin, self.rt, 7, (0, 255, 255), -1)
        cv2.circle(skin, self.tt, 7, (255, 0, 255), -1)
        cv2.circle(skin, self.bt, 7, (238, 103, 122), -1)

    def show(self, frame, skin):

        #对frame 原始图像
        # 显示提示语
        cv2.putText(frame, "Option: ", (self.fx, self.fy), self.font, self.size, (0, 255, 0))  # 标注字体
        cv2.putText(frame, "s-'save image'", (self.fx, self.fy + 1 * self.fh), self.font, self.size,
                (0, 255, 0))  # 标注字体
        #
        cv2.putText(frame, "q-'quit'", (self.fx, self.fy + 2 * self.fh), self.font, self.size, (0, 255, 0))  # 标注字体

        text = "position : (" + str(self.lt[0]) + ", " + str(self.lt[1]) +")"
        cv2.putText(frame, text, (self.fx, self.fy + 3 * self.fh), self.font, self.size,
                (0, 255, 0))  # 标注字体

        #显示分离出来的图像的质心
        cv2.circle(frame, self.center, 5, self.green, -1)
        cv2.circle(frame, self.lt, 5, self.red, -1)
        """
        #绘制分离出来图形的轮廓
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        mask = cv2.drawContours(mask, [self.contounr], -1, 255, -1)

        # 计算分离出来图像的最大最小值,及其索引 ,但是我们这里使用极点来追逐,因此不用最大最小值
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, mask=mask)
        text = "position : (" + str(max_loc[0]) + ", " + str(max_loc[1]) + ")"
        cv2.putText(frame, text  , (self.fx, self.fy + 3 * self.fh), self.font, self.size, (0, 255, 0))  # 标注字体
        """


        #值分离出来的图像上显示 最左侧极点
        cv2.circle(skin, self.center, 5, self.green, -1)
        cv2.circle(skin, self.lt, 5, self.red, -1)
        # 绘制轮廓
        cv2.drawContours(frame, [self.contounr], 0, self.blue,3)


    def show_in_one(self, frame, skin):
        self.show(frame, skin)
        self.pointPose(frame, skin)
        #
        self.drawCountours(skin, frame)
        #
        hstack1 = np.hstack((frame,skin))

        # 这个图像是单通道，而另外两个是三通道，要一起显示需要把通道数量变为相同的
        ret = cv2.cvtColor(self.binaryimage, cv2.COLOR_GRAY2BGR)
        black = cv2.cvtColor(self.black, cv2.COLOR_GRAY2BGR)
        hstack2 = np.hstack((ret, black))

        #合并为一个，便于一起显示
        vstack3 = np.vstack((hstack1, hstack2))
        cv2.imshow("all_in", vstack3)


    def run(self):
        # cap = cv2.VideoCapture(0)  # 我们这里默认打开0号摄像头
        while True:
            # if True != cap.isOpened():
            #     print("No camera Open, please checked")
            #     return
            # # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
            # ret, frame = cap.read()
            frame = cv2.imread("hand06.jpg", 1)
            # frame = cv2.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转

            # 显示
            self.img = frame
            #
            skin = self.binaryMask(frame)
            self.show_in_one(frame, skin)

            # self.trackHand()

            key = cv2.waitKey(1) & 0xFF  # 等待键盘输入，
            if key == ord('b'):  # 将ROI显示为二值模式
                # binaryMode = not binaryMode
                binaryMode = True
                print("Binary Threshold filter active")
            elif key == ord('r'):  # RGB模式
                binaryMode = False

            if key == ord('q') :
                break

        # 最后记得释放捕捉
        # cap.release()
        cv2.destroyAllWindows()


ex = Extract_With_Fourier_Descriptors()
ex.run()
