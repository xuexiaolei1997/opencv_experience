# 哈里斯角检测  cv.cornerHarris(), cv.cornerSubPix()
# SubPixel精度的转角 cv.cornerSubPix

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

filename = '2.jpg'
img = cv.imread(filename)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
# result用于标记角点，并不重要
dst = cv.dilate(dst, None)
# 最佳值的阈值，它可能因图像而异。
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv.imshow('dst', img)

# ==============================

ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)
# 寻找质心
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# 定义停止和完善拐角的条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids), (5, 5), (-1, -1), criteria)
# 绘制
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]
img[res[:, 3], res[:, 2]] = [0, 255, 0]
cv.imshow('res', img)
cv.waitKey(0)
cv.destroyAllWindows()

