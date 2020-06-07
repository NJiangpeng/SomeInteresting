import numpy as np
import cv2
import os

def getCapture():
    cap = cv2.VideoCapture(0)
    while True:
        # get a frame
        ret, frame = cap.read()

        # our operation on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def getVideo():
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    witdh = 640
    hight = 480
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (witdh, hight))
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)
            i += 1
             # write the flipped frame
            out.write(frame)
            name = "./pen/book_pen_" + str(i) +".png"
            cv2.imwrite(name, frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def changename(path):
    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)
    print(fileList)
    print(len(fileList))

    for i, name in enumerate(fileList):
        newname = "book_pen_" + str(i)  +".png"
        os.rename(path + name, path +newname)
    # for i in range(len(fileList)):
    #     # 设置旧文件名（就是路径+文件名）
    #     oldname = path + os.sep + fileList[i]  # os.sep添加系统分隔符
    #
    #     # 设置新文件名
    #     p = path + os.sep
    #     newname = p + name + "_" + str(i) + '.png'
    #
    #     os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    #     print(oldname, '======>', newname)
    print("Done")

# getVideo()
path = "G:/testFinger/pen/"
changename(path)