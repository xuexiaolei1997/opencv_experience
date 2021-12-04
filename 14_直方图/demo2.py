import cv2 as cv
import numpy as np

img = cv.imread("3.jpg", 0)
# 直方图均衡
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cll = clahe.apply(img)
cv.imshow("img", img)
cv.imshow("res", cll)
cv.waitKey(0)
cv.destroyAllWindows()

