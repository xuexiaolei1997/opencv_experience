import cv2 as cv

img1 = cv.imread("2.jpg", cv.IMREAD_GRAYSCALE)
backSub = cv.createBackgroundSubtractorMOG2()

fgMask = backSub.apply(img1)
cv.rectangle(img1, (10, 2), (100, 20), (255, 255, 255), -1)
cv.putText(img1, "test", (15, 15),
           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
cv.imshow("img", img1)
cv.imshow("fgMask", fgMask)
cv.waitKey(0)
cv.destroyAllWindows()
