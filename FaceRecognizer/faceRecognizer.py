# -*- coding: utf-8 -*-
# =========================================================
# @Author:
#                             _                  _        _ 
#                            (_)                | |      (_)
# __  __ _   _   ___   __  __ _   __ _   ___    | |  ___  _ 
# \ \/ /| | | | / _ \  \ \/ /| | / _` | / _ \   | | / _ \| |
#  >  < | |_| ||  __/   >  < | || (_| || (_) |  | ||  __/| |
# /_/\_\ \__,_| \___|  /_/\_\|_| \__,_| \___/   |_| \___||_|
# @Time: 2021/10/6 18:11
# =========================================================
import os
import cv2 as cv

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

names = []

warningtime = 0


def name():
    path = './data/'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = os.path.split(imagePath)[1].split('.')[1]
        names.append(name)


def face_detect_func(i):
    gray = cv.cvtColor(i, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier(
        'D:/Software/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray, 1.1, 5, cv.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    for x, y, w, h in face:
        cv.rectangle(i, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        ids, confidence = recognizer.predict(gray[y:y + h, x: x + w])
        if confidence > 80:
            global warningtime
            warningtime += 1
            if warningtime > 100:
                # warning()
                warningtime = 0
            cv.putText(i, 'unknow', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            print("id:{}, name:{}".format(ids, names[ids - 1]))
            cv.putText(i, str(names[ids - 1]), (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv.imshow("res", i)


name()
cap = cv.VideoCapture(0)
num = 0
while cap.isOpened():
    ret_flag, Vshow = cap.read()
    face_detect_func(Vshow)
    # cv.imshow("Capture Test", Vshow)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        cv.imwrite("{}.jpg".format(num), Vshow)
        num += 1
    elif k == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
