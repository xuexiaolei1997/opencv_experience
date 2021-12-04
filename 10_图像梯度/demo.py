# OpenCV提供三种类型的梯度滤波器或高通滤波器，即Sobel，Scharr和Laplacian。

# Sobel算子是高斯平滑加微分运算的联合运算，因此它更抗噪声。逆可以指定要采用的导数方向，
# 垂直或水平（分别通过参数yorder和xorder）。逆还可以通过参数ksize指定内核的大小。

# Laplacian 算子
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("1.jpg", 0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("1", img)

laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]),  plt.yticks([])
plt.show()

