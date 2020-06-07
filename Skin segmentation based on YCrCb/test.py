import cv2

cap = cv2.VideoCapture(0)

i= 0
while(1):
    # 获得图片
    ret, frame = cap.read()
    # 展示图片
    cv2.imshow("capture", frame)

    k = cv2.waitKey(1)
    if k == ord('s'):
        name = "pen_" + str(i) +".jpg"
        i += 1
        cv2.imwrite(name, frame)
    if k == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()
