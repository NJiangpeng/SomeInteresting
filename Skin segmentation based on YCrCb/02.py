#####
# 一、高斯滤波
####
# 导入需要的包
import cv2
import os
import numpy as np
import time


class Deal_With_GaussFilter():
    def __init__(self):
        # 设置一些常用的一些参数
        # 显示的字体 大小 初始位置等
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 正常大小无衬线字体
        self.size = 0.5
        self.fx = 10
        self.fy = 355
        self.fh = 18
        # ROI框的显示位置
        self.x0 = 300
        self.y0 = 100
        # 录制的手势图片大小
        self.width = 300
        self.height = 300
        # 每个手势录制的样本数
        self.numofsamples = 300
        self.counter = 0  # 计数器，记录已经录制多少图片了
        # 存储地址和初始文件夹名称
        self.floder = 'finger'
        self.path = './test/'
        self.imgname = "gesture"
        self.record = False


        # 标识符 bool类型用来表示某些需要不断变化的状态
        self.binaryMode = False  # 是否将ROI显示为而至二值模式
        self.saveImg = False  # 是否需要保存图片

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
        # # 显示方框
        # cv2.rectangle(frame, (self.x0, self.y0), (self.x0 + self.width, self.y0 + self.height), (0, 255, 0))
        # # 提取ROI像素
        # roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width]  #
        roi = frame
        # 高斯滤波处理
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 高斯模糊 斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
        # 高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大
        sz = 5
        blur = cv2.GaussianBlur(gray, (sz, sz), 2)  # 高斯模糊，给出高斯模糊矩阵和标准差

        # 当同一幅图像上的不同部分的具有不同亮度时。这种情况下我们需要采用自适应阈值
        # 参数： src 指原图像，原图像应该是灰度图。 x ：指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
        #  adaptive_method  指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
        # block_size           指用来计算阈值的象素邻域大小: 3, 5, 7, ..
        #   param1           指与方法有关的参数    #
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # ret还是bool类型

        # "这里可以插入代码调用网络"
        # 二值化处理
        kernel = np.ones((sz, sz), np.uint8)  # 设置卷积核
        erosion = cv2.erode(res, kernel)  # 腐蚀操作 开运算：先腐蚀后膨胀，去除孤立的小点，毛刺
        cv2.imshow("erosion", erosion)
        erosion = cv2.medianBlur(erosion, 5)
        dilation = cv2.dilate(erosion, kernel)  # 膨胀操作 闭运算：先膨胀后腐蚀，填平小孔，弥合小裂缝
        cv2.imshow("dilation", dilation)
        # 轮廓提取
        binaryimg = cv2.Canny(res, 50, 200)  # 二值化，canny检测
        h = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
        contours = h[0]  # 提取轮廓
        ret = np.ones(res.shape, np.uint8)  # 创建黑色幕布
        cv2.drawContours(ret, contours, -1, (255, 255, 255), 1)  # 绘制白色轮廓
        cv2.imshow("ret", ret)

        # 保存手势
        if self.saveImg == True and self.binaryMode == True:
            self.saveROI(res)
        elif self.saveImg == True and self.binaryMode == False:
            self.saveROI(roi)
        return res

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
            cv2.putText(frame, "s-'new gestures(twice)'", (self.fx, self.fy + 2 * self.fh), self.font, self.size,
                        (0, 255, 0))  # 标注字体
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

            # 展示处理之后的视频帧
            cv2.imshow('frame', frame)
            # if (self.binaryMode):
            #     cv2.imshow('ROI', roi)
            # else:
            #     cv2.imshow("ROI", frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width])

        # 最后记得释放捕捉
        cap.release()
        cv2.destroyAllWindows()




st = Deal_With_GaussFilter()
st.run()
