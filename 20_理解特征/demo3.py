# SIFT尺度不变特征变换
import numpy as np
import cv2 as cv
img = cv.imread('2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imshow("res", img)
cv.waitKey(0)
cv.destroyAllWindows()
