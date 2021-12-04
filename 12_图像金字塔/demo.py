# 高斯金字塔 和 拉普拉斯金字塔

import cv2 as cv
img = cv.imread("1.jpg", 1)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

lower_reso = cv.pyrDown(img)
higher_reso = cv.pyrUp(img)
higher_reso2 = cv.pyrUp(lower_reso)
cv.imshow("0", img)
cv.imshow("1", lower_reso)
cv.imshow("2", higher_reso)
cv.imshow("3", higher_reso2)
cv.waitKey(0)
