import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
mask = np.zeros(img.shape[:2], np.uint8)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)
# cv.rectangle(img, rect[:2], rect[2:], (0, 0, 255), 1, 1)
# cv.imshow("a", img)
cv.grabCut(img, mask, rect, bgModel, fgModel, 5, cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

img = img * mask2[:, :, np.newaxis]
plt.imshow(img)
plt.show()
