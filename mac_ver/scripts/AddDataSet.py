#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
# from scipy import ndimage
import sys
# import os
from time import time
import utils as ut
import svm
from sklearn.externals import joblib

#############################################################################
if len(sys.argv) != 2:
    print "Please input the profile name!"
    print "example: python AddDataSet.py guoyang"
    sys.exit(0)

profile_folder_path = ut.create_profile_in_database(sys.argv[1])
##############################################################################

face_cascade = cv2.CascadeClassifier("../classifier/haarcascade_frontalface_alt.xml")
if face_cascade is None:
    print "Load face CascadeClassifier failed!"
    sys.exit(0)

webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()

################################################################################

FACE_DIM = (100, 100)
save_num = 0
while ret:
    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]:
        break
    start = time()
    origin = frame
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    faces = face_cascade.detectMultiScale(
        origin,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=0
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(origin, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y+h, x:x+w]
        cv2.imshow("face", face)
        face_to_save = cv2.resize(face, FACE_DIM, interpolation=cv2.INTER_AREA)
        if key == ord('p'):
            print 'add '+str(save_num)+' picture'
            face_name = profile_folder_path+str(save_num)+'.png'
            cv2.imwrite(face_name, face_to_save)
            save_num = save_num + 1

    cv2.imshow("video", origin)
    ret, frame = webcam.read()

webcam.release()
cv2.destroyAllWindows()