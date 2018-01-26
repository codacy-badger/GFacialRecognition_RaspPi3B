#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auther: Guo Yang <guoyang@hzau.edu.cn>

This file is part of ImgProcessing A Project.

Summary: real_time facial recognition ( suitable for RasPi now )

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

face_profile_names = ut.get_profile_names('../face_profiles/')

clf = joblib.load('../model/svm.pkl')
pca = joblib.load('../model/pca.pkl')
face_cascade = cv2.CascadeClassifier("../classifier/haarcascade_frontalface_alt.xml")
if face_cascade is None:
    print "Load face CascadeClassifier failed!"
    sys.exit(0)

FACE_DIM = (100, 100) 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (512, 400)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(512, 400))
 
# allow the camera to warmup
time.sleep(2)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	origin = frame.array
 
	# show the frame
	#cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	start = time.time()
	#origin = frame
	gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray,gray)

	faces = face_cascade.detectMultiScale(
		origin,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		# flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		flags=0
	)
	for (x, y, w, h) in faces:
		cv2.rectangle(origin, (x, y), (x+w, y+h), (0, 255, 0), 2)
		face = gray[y:y+h, x:x+w]
		cv2.equalizeHist(face, face)

		face_to_predict = cv2.resize(face, FACE_DIM, interpolation=cv2.INTER_AREA)
		name_predict = svm.predict_single(clf, pca, face_to_predict, face_profile_names)
		cv2.putText(origin, name_predict, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))

	# --------Calculate fps---------------------------------------------------
	fps = 1.0/(time.time() - start)
	FPS_info = "FPS: "+"{:.2f}".format(fps)
	cv2.putText(origin, FPS_info, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
	cv2.imshow("Knock Knock", origin)

cv2.destroyAllWindows()
