#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from scipy import ndimage
from time import time
#import matplotlib.pyplot as plt
import utils as ut
import svm
import sys
import logging
import warnings
from sklearn.externals import joblib

FACE_DIM = (100,100) # height = 100, weight = 100
face_profile_data, face_profile_name_index, face_profile_names  = ut.load_training_data("../face_profiles/")
print "\n", face_profile_name_index.shape[0], " samples from ", len(face_profile_names), " people are loaded"
clf, pca = svm.build_SVC(face_profile_data, face_profile_name_index, FACE_DIM)

joblib.dump(clf,'../model/svm.pkl')
joblib.dump(pca,'../model/pca.pkl')



