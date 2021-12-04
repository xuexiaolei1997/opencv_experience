# Shi-tomas拐角检测器和益于跟踪的特征 cv.goodFeaturesToTrack

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray, 10, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)
plt.imshow(img)
plt.show()
