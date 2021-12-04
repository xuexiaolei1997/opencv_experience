# BRIEF(二进制的鲁棒独立基本特征)
# Pass 版本更改

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('1.jpg', 0)
# 初始化FAST检测器
star = cv.xfeatures2d.StarDetector_create()
# 初始化BRIEF提取器
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# 找到STAR的关键点
kp = star.detect(img, None)
# 计算BRIEF的描述符
kp,  des = brief.compute(img,  kp)
print(brief.descriptorSize())
print(des.shape)
