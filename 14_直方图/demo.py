import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('1.jpg')

# cv
hist = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

# numpy
# 1
# hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# 2
hist = np.bincount(img.ravel(), minlength=256)
plt.plot(hist)
plt.show()

# matplotlib
# plt.hist(img.ravel(), 256, [0, 256])
color = ('b', 'g', 'r')
plt.figure()
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    # plt.xlim([0, 256])
plt.show()
