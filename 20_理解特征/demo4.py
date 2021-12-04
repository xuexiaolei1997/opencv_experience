# SURF擅长处理具有模糊和旋转的图像，但不擅长处理视点变化和照明变化
# Pass 版本更改

import cv2 as cv
import numpy as np
img = cv.imread('3.jpg', 0)
surf = cv.SURF_create(400)
kp, des = surf.detectAndCompute(img, None)


