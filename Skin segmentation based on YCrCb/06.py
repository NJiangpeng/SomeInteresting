
import cv2
import os
import numpy as np
import time
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
        self.center_x = 0
        self.center_y = 0
        #
        self.lt =(1)
        self.rt = (1)
        self.tt = (1)
        self.bt = (1)

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
        # contour = find_contours(Laplacian)
        contour = self.find_contours(Laplacian) # 提取轮廓点坐标
        contour_array = contour[0][:, 0, :]  # 注意这里只保留区域面积最大的轮廓点坐标

        #
        cnt = contour[0]
        self.lt = tuple(cnt[cnt[:, :, 0].argmin()][0])
        self.rt = tuple(cnt[cnt[:, :, 0].argmax()][0])
        self.tt = tuple(cnt[cnt[:, :, 1].argmin()][0])
        self.bt = tuple(cnt[cnt[:, :, 1].argmax()][0])

        #
        M = cv2.moments(contour_array)

        self.center_x = int(M['m10'] / M['m00'])
        self.center_y = int(M['m01'] / M['m00'])
        # print(approx)



        ret_np = np.ones(dst.shape, np.uint8)  # 创建黑色幕布
        ret = cv2.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # 绘制白色轮廓
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



        return  contour_array ,ret, descirptor_in_use, black

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

        # ret, fourier_desciptor = fourierDesciptor(ros) #获取傅里叶描述算子的轮廓点
        contour0, ret, fourier_desciptor, black = self.fourierDesciptor(skin)  # 获取傅里叶描述算子的轮廓点

        if self.saveImg == True and self.binaryMode == True:
            self.saveROI(skin)
        elif self.saveImg == True and self.binaryMode == False:
            self.saveROI(roi)
        return contour0,skin, black, ret

    def run(self):
        cap = cv2.VideoCapture(0)  # 我们这里默认打开0号摄像头
        while True:
            if True != cap.isOpened():
                print("No camera checked")
                return
            # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
            ret, frame = cap.read()
            frame = cv2.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转


            contour0, skin, black, ret = self.binaryMask(frame)

            # 显示提示语
            # cv2.putText(frame, "Option: ", (self.fx, self.fy), self.font, self.size, (0, 255, 0))  # 标注字体
            # cv2.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (self.fx, self.fy + self.fh), self.font, self.size,
            #         (0, 255, 0))  # 标注字体
            # cv2.putText(frame, "p-'prediction mode'", (self.fx, self.fy + 2 * self.fh), self.font, self.size,
            #         (0, 255, 0))  # 标注字体
            # cv2.putText(frame, "s-'new gestures(twice)'", (self.fx, self.fy + 3 * self.fh), self.font, self.size,
            #         (0, 255, 0))  # 标注字体
            cv2.putText(frame, "q-'quit'", (self.fx, self.fy + 4 * self.fh), self.font, self.size, (0, 255, 0))  # 标注字体

            text = "position : (" + str(self.lt[0]) + ", " + str(self.lt[1]) +")"
            cv2.putText(frame, text, (self.fx, self.fy + 3 * self.fh), self.font, self.size,
                    (0, 255, 0))  # 标注字体
            cv2.circle(frame, (self.center_x, self.center_y), 5, (255, 0, 255), -1)


            #

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, np.uint8)
            mask = cv2.drawContours(mask, [contour0], -1, 255, -1)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, mask=mask)



            text = "fingertip position : (" + str(max_loc[0]) + ", " + str(max_loc[1]) + ")"
            # cv2.putText(frame, text  , (self.fx, self.fy + 3 * self.fh), self.font, self.size, (0, 255, 0))  # 标注字体

            cv2.circle(skin, self.lt, 5,(106, 106,255), -1)
            cv2.circle(frame, self.lt, 5, (106, 106,255), -1)
            # cv2.circle(frame, self.rt, 7, (0,   255, 255), -1)
            # cv2.circle(frame, self.tt, 7, (255, 0, 255), -1)
            # cv2.circle(frame, self.bt, 7, (238, 103, 122), -1)

            cv2.line(frame, (self.center_x, self.center_y), self.lt, (106, 106,255), 2)
            # cv2.line(frame, (self.center_x, self.center_y), self.rt, (0,   255, 255), 2)
            # cv2.line(frame, (self.center_x, self.center_y), self.tt, (255, 0, 255), 2)
            # cv2.line(frame, (self.center_x, self.center_y), self.bt, (238, 103, 122), 2)




            # cv2.drawContours(frame, [contour0], 0, (0,  0, 255), 3)

            hstack1 = np.hstack((frame, skin))

            ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
            black = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
            hstack2 = np.hstack((ret, black))

            # print(ret.shape)
            # print((black.shape))
            vstack3 = np.vstack((hstack1, hstack2))
            cv2.imshow("vs", vstack3)
            # 展示处理之后的视频帧
            # cv2.imshow('frame', frame)
            # cv2.imshow("in_one", hstack1)
            # cv2.imshow("in_two", hstack2)


            key = cv2.waitKey(1) & 0xFF  # 等待键盘输入，
            if key == ord('b'):  # 将ROI显示为二值模式
                # binaryMode = not binaryMode
                binaryMode = True
                print("Binary Threshold filter active")
            elif key == ord('r'):  # RGB模式
                binaryMode = False

            #     if key == ord('i'):  # 调整ROI框
            #         y0 = self.y0 - 5
            # elif key == ord('k'):
            #     y0 = self.y0 + 5
            # elif key == ord('j'):
            #     x0 = self.x0 - 5
            # elif key == ord('l'):
            #     x0 = self.x0 + 5

            # if key == ord('p'):
            #     """调用模型开始预测"""
            #     print("using CNN to predict")
            if key == ord('q'):
                break


            if key == ord('s'):
                """录制新的手势（训练集）"""
                # saveImg = not saveImg # True
                if self.gesturename != '':
                    saveImg = True
                else:
                    print("Enter a gesture group name first, by enter press 'n'! ")
                    saveImg = False
            # elif key == ord('n'):
            #     # 开始录制新手势
            #     # 首先输入文件夹名字
            #     self.gesturename = (input("enter the gesture folder name: "))
            #     os.makedirs(gesturename)
            #     self.path = "./" + gesturename + "/"  # 生成文件夹的地址  用来存放录制的手势

            # if (self.binaryMode):
            #     cv2.imshow('ROI', roi)
            # else:
            #     cv2.imshow("ROI", frame[self.y0: self.y0 + self.height, self.x0: self.x0 + self.width])

        # 最后记得释放捕捉
        cap.release()
        cv2.destroyAllWindows()

ex = Extract_With_Fourier_Descriptors()
ex.run()