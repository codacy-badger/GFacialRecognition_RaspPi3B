#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auther: Guo Yang <guoyang@hzau.edu.cn>

This file is part of ImgProcessing A Project.

Summary: register a new face ( suitable for RasPi now )

"""

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from scipy import ndimage
import sys
import os
import utils as ut
import svm
from sklearn.externals import joblib

#############################################################################
if len(sys.argv) != 2:
    print "Please input the profile name!"
    print "example: python AddDataSet.py david"
    sys.exit(0)

profile_folder_path = ut.create_profile_in_database(sys.argv[1])
##############################################################################

face_cascade = cv2.CascadeClassifier("../classifier/haarcascade_frontalface_alt.xml")
if face_cascade is None:
    print "Load face CascadeClassifier failed!"
    sys.exit(0)

################################################################################

FACE_DIM = (100, 100)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (512, 400)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(512, 400))
 
# allow the camera to warmup
time.sleep(2)

save_num = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	origin = frame.array
	key1 = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	if key1 in [27, ord('Q'), ord('q')]:
    		break
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
		if key1 in [ ord('p'), ord('P')]:
			print 'add '+str(save_num)+' picture'
			face_name = profile_folder_path+str(save_num)+'.png'
			cv2.imwrite(face_name, face_to_save)
			save_num = save_num + 1
	cv2.imshow("Register", origin)

cv2.destroyAllWindows()
