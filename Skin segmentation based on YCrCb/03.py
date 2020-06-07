"""
二、YCrCb肤色检测
（2.1）思路
加载图像（opencv，截图保存saveROI）
肤色检测（YCrCb颜色空间的Cr分量+Otsu法阈值分割算法）
图像去噪（numpy二值化处理）
轮廓提取（canny检测，cv2.findContours）
绘制轮廓（cv2.drawContours）
（2.2）源码

"""

# 导入需要的包
import cv2
import os
import numpy as np
import time


class YCRCB_Skin():
    def __init__(self):
        # 设置一些常用的一些参数
        # 显示的字体 大小 初始位置等
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 正常大小无衬线字体
        self.size = 0.5
        self.fx = 10
        self.fy = 355
        self.fh = 18
        # ROI框的显示位置
        self.x0 = 0
        self.y0 = 0
        # 录制的手势图片大小
        self.width = 640
        self.height = 480
        # 每个手势录制的样本数
        self.numofsamples = 300
        self.counter = 0  # 计数器，记录已经录制多少图片了
        # 存储地址和初始文件夹名称
        self.gesturename = ''
        self. path = ''
        # 标识符 bool类型用来表示某些需要不断变化的状态
        self.binaryMode = False  # 是否将ROI显示为而至二值模式
        self.saveImg = False  # 是否需要保存图片

        #
        self.lt = (1)
        self.rt = (1)
        self.tt = (1)
        self.bt = (1)
        self.center_x = 0
        self.counter_y = 0

    #YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
    def skinMask(self, roi):
        YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
        (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
        skin = cv2.bitwise_and(roi, roi, mask=skin)
        return skin

    # 保存ROI图像
    def saveROI(self, img):
        if self.saveImg != True:
            return

        if self.counter > self.numofsamples:
            # 恢复到初始值，以便后面继续录制手势
            self.counter = 0

        self.counter += 1
        name = self.imgname + "_" + str(self.counter)  # 给录制的手势命名
        print("Saving img: ", name)

        cv2.imwrite(self.path + name + '.png', img)  # 写入文件
        time.sleep(0.05)

    # 显示ROI为二值模式
    def binaryMask(self, frame):
        # cv2.rectangle(frame, (self.x0, self.y0), (self.x0 + self.width, self.y0 + self.height), (0, 255, 0))  # 画出截取的手势框图
        # roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width]  # 获取手势框图
        roi = frame
        # cv2.imshow("roi", roi)  # 显示手势框图
        # 肤色检测
        skin = self.skinMask(roi)  # 进行肤色检测
        # skin = cv2.medianBlur(skin, 3)
        cv2.imshow("skin", skin)  # 显示肤色检测后的图像

        "这里可以插入代码调用网络"
        # 二值化处理
        ksize = 3
        kernel = np.ones(( ksize,  ksize), np.uint8)  # 设置卷积核
        erosion = cv2.erode(skin, kernel)  # 腐蚀操作 开运算：先腐蚀后膨胀，去除孤立的小点，毛刺
        # cv2.imshow("erosion", erosion)
        dilation = cv2.dilate(erosion, kernel)  # 膨胀操作 闭运算：先膨胀后腐蚀，填平小孔，弥合小裂缝
        # cv2.imshow("dilation", dilation)
        # 轮廓提取
        binaryimg = cv2.Canny(skin, 50, 200)  # 二值化，canny检测
        contours = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
        contour_array = contours[0]  # 提取轮廓
        img = np.ones(skin.shape, np.uint8)  # 创建黑色幕布
        cv2.drawContours(img, contour_array, -1, (255, 255, 255), 1)  # 绘制白色轮廓
        # img = cv2.medianBlur(img, 3)
        cv2.imshow("contour", img)

        ###
        coutour = sorted(contours[0], key=cv2.contourArea, reverse=True) # 对一系列轮廓点坐标按它们围成的区域面积进行排序
        cnt = coutour[0]
        self.lt = tuple(cnt[cnt[:, :, 0].argmin()][0])
        self.rt = tuple(cnt[cnt[:, :, 0].argmax()][0])
        self.tt = tuple(cnt[cnt[:, :, 1].argmin()][0])
        self.bt = tuple(cnt[cnt[:, :, 1].argmax()][0])
        contour_array = coutour[0][:, 0, :]  # 注意这里只保留区域面积最大的轮廓点坐标
        M = cv2.moments(contour_array)

        self.center_x = int(M['m10'] / M['m00'])
        self.center_y = int(M['m01'] / M['m00'])
        ###

        #
        # 保存手势
        if self.saveImg == True and self.binaryMode == True:
            self.saveROI(skin)
        elif self.saveImg == True and self.binaryMode == False:
            self.saveROI(roi)
        #
        return skin

    def run(self):
        # 创建一个视频捕捉对象
        cap = cv2.VideoCapture(0)  # 0为（笔记本）内置摄像头
        while (True):
            # 读帧
            ret, frame = cap.read()  # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
            # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
            frame = cv2.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转
            # 显示ROI区域 # 调用函数
            roi = self.binaryMask(frame)
            # 显示提示语
            cv2.putText(frame, "Option: ", (self.fx, self.fy), self.font, self.size, (0, 255, 0))  # 标注字体
            cv2.putText(frame, "c-'chang mode'(binary|RGB) ", (self.fx, self.fy + self.fh), self.font, self.size,
                        (0, 255, 0))  # 标注字体
            # cv2.putText(frame, "s-'new gestures(twice)'", (self.fx, self.fy + 2 * self.fh), self.font, self.size,
            #             (0, 255, 0))  # 标注字体
            cv2.putText(frame, "q-'quit'", (self.fx, self.fy + 3 * self.fh), self.font, self.size, (0, 255, 0))  # 标注字体



            key = cv2.waitKey(1) & 0xFF  # 等待键盘输入，


            if key == ord('c'):
                # print("chang mode")
                self.binaryMode = not self.binaryMode


            if key == ord('q'):
                break

            if key == ord('s'):
                self.saveImg = not self.saveImg #用来标志是否存储照片
                if self.saveImg == True:
                    # print("enter a folder to save image!")
                    self.floder = (input("enter folder name : "))
                    if self.floder == "":
                        self.floder = "test"
                    if not os.path.exists(self.floder):
                        os.mkdir(self.floder)
                    self.path = "./" + self.floder + "/"

                    # print("enter image name!")
                    self.imgname = ((input("enter image name: ")))
                    if self.imgname == "":
                        self.imgname = "test"


            #
            # cv2.circle(frame, (self.center_x, self.center_y), 5, (255, 0, 255), -1)
            cv2.circle(frame, self.lt, 5, (106, 106, 255), -1)
            # cv2.circle(frame, self.rt, 7, (0,   255, 255), -1)
            # cv2.circle(frame, self.tt, 7, (255, 0, 255), -1)
            # cv2.circle(frame, self.bt, 7, (238, 103, 122), -1)

            # cv2.line(frame, (self.center_x, self.center_y), self.lt, (106, 106, 255), 2)
            # cv2.line(frame, (self.center_x, self.center_y), self.rt, (0,   255, 255), 2)
            # cv2.line(frame, (self.center_x, self.center_y), self.tt, (255, 0, 255), 2)
            # cv2.line(frame, (self.center_x, self.center_y), self.bt, (238, 103, 122), 2)
            #

            # 展示处理之后的视频帧
            cv2.imshow('frame', frame)
            if (self.binaryMode):
                cv2.imshow('ROI', roi)
            else:
                cv2.imshow("ROI", frame[self.y0: self.y0 + self.height, self.x0: self.x0 + self.width])

        # 最后记得释放捕捉
        cap.release()
        cv2.destroyAllWindows()


st = YCRCB_Skin()
st.run()