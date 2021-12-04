# -*- coding: utf-8 -*-
# =========================================================
# @Author:
#                             _                  _        _ 
#                            (_)                | |      (_)
# __  __ _   _   ___   __  __ _   __ _   ___    | |  ___  _ 
# \ \/ /| | | | / _ \  \ \/ /| | / _` | / _ \   | | / _ \| |
#  >  < | |_| ||  __/   >  < | || (_| || (_) |  | ||  __/| |
# /_/\_\ \__,_| \___|  /_/\_\|_| \__,_| \___/   |_| \___||_|
# @Time: 2021/10/6 17:10
# =========================================================
import os

import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def getImageAndLabels(path):
    facesSamples = []
    ids = []
    names = []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face_detector = cv.CascadeClassifier(
        'D:/Software/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_image, 'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        id = int(os.path.split(imagePath)[1].split('.')[0])
        name = os.path.split(imagePath)[1].split('.')[1]
        for x, y, w, h in faces:
            ids.append(id)
            names.append(name)
            facesSamples.append(img_numpy[y:y + h, x:x + w])
            # plt.imshow(img_numpy[y:y + h, x:x + w])
            # plt.show()
    print('ids:', ids)
    print('names:', names)
    print('fs:', facesSamples)
    return facesSamples, ids


if __name__ == '__main__':
    path = './data/'
    faces, ids = getImageAndLabels(path)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
