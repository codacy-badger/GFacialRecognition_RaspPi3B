"""
Auther: Guo Yang <guoyang@webmail.hzau.edu.cn>

This file is part of <<ImgProcessing A>> Project.

Summary: SVM methods using Scikit

"""

import cv2
import os
import tkMessageBox
import numpy as np
from scipy import ndimage
from time import time
import warnings
from led_control import knockknock_mode, train_mode

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.cross_validation import train_test_split

from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import utils as ut

def build_SVC(face_profile_data, face_profile_name_index, face_dim):
    """
    Build the SVM classification modle using the face_profile_data matrix (numOfFace X numOfPixel) and face_profile_name_index array, face_dim is a tuple of the dimension of each image(h,w) Returns the SVM classification modle
    Parameters
    ----------
    face_profile_data : ndarray (number_of_images_in_face_profiles, width * height of the image)
        The pca that contains the top eigenvectors extracted using approximated Singular Value Decomposition of the data

    face_profile_name_index : ndarray
        The name corresponding to the face profile is encoded in its index

    face_dim : tuple (int, int)
        The dimension of the face data is reshaped to

    Returns
    -------
    clf : theano object
        The trained SVM classification model

    pca : theano ojbect
        The pca that contains the top 150 eigenvectors extracted using approximated Singular Value Decomposition of the data

    """

    X = face_profile_data
    y = face_profile_name_index

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150 # maximum number of components to keep
    train_mode( )
    print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components,  face_dim[0], face_dim[1]))

    # This portion of the code is used if the data is scarce, it uses the number 
    # of imputs as the number of features
    # pca = RandomizedPCA(n_components=None, whiten=True).fit(X_train)
    # eigenfaces = pca.components_.reshape((pca.components_.shape[0], face_dim[0], face_dim[1]))

    print("\nProjecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train a SVM classification model

    print("\nFitting the classifier to the training set")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

    # Best Estimator found using Radial Basis Function Kernal:
    clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


    clf = clf.fit(X_train_pca, y_train)
    # print("\nBest estimator found by grid search:")
    # print(clf.best_estimator_)

    ###############################################################################
    # Quantitative evaluation of the model quality on the test set
    print("\nPredicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("\nPrediction took %s per sample on average" % ((time() - t0)/y_pred.shape[0]*1.0))

    # print "predicated names: ", y_pred
    # print "actual names: ", y_test
    error_rate = errorRate(y_pred, y_test)
    print ("\nTest Error Rate: %0.4f %%" % (error_rate * 100))
    print ("Test Recognition Rate: %0.4f %%" % ((1.0 - error_rate) * 100))

    tkMessageBox.showinfo(title='train result', message="\nTest Error Rate: %0.4f \nTest Recognition Rate: %0.4f %%" % (error_rate * 100, (1.0 - error_rate) * 100))

    return clf, pca


def predict(clf, pca, img, face_profile_names):
    """
    Predict the name of the supplied image from the list of face profile names

    Parameters
    ----------
    clf: theano object
        The trained svm classifier 

    pca: theano object
        The pca that contains the top eigenvectors extracted using approximated Singular Value Decomposition of the data

    img: ndarray
        The input image for prediction

    face_profile_names: list
       The names corresponding to the face profiles
    Returns
    -------
    name : string
        The predicated name

    """

    img = img.ravel()
    # Apply dimentionality reduction on img, img is projected on the first principal components previous extracted from the Yale Extended dataset B.
    principle_components = pca.transform(img)
    pred = clf.predict(principle_components)
    name = face_profile_names[pred]

    return name

def errorRate(pred, actual):
    """
    Calculate name prediction error rate

    Parameters
    ----------
    pred: ndarray (1, number_of_images_in_face_profiles)
        The predicated names of the test dataset

    actual: ndarray (1, number_of_images_in_face_profiles)
        The actual names of the test dataset

    Returns
    -------
    error_rate: float
        The calcualted error rate

    """
    if pred.shape != actual.shape: return None
    error_rate = np.count_nonzero(pred - actual)/float(pred.shape[0])
    return error_rate

def printresult(pred,actual,names):
    """
        Print the differnce between predict and actual results

        Parameters
        ----------
        pred: ndarray (1, number_of_images_in_face_profiles)
            The predicated names of the test dataset

        actual: ndarray (1, number_of_images_in_face_profiles)
            The actual names of the test dataset
    """
    if pred.shape != actual.shape: return None
    for i in xrange(pred.shape[0]):
        print "acutal is %s prediction is %s" %(names[actual[i]], names[pred[i]])


def predict_single(clf, pca, img, face_profile_names):
    img = img.ravel()
    # Apply dimentionality reduction on img, img is projected on the first principal components previous extracted from the Yale Extended dataset B.
    principle_components = pca.transform(img.reshape(1, -1))  # +++++++++++++++++++++++
    # print principle_components
    proba = clf.predict_proba(principle_components)  # probability # +++++++++++++++++++++++
    pred = clf.predict(principle_components)
    print proba[0][pred]
    if proba[0][pred] > 0.7:
        name = face_profile_names[int(pred)]
	print name
        knockknock_mode(1)
    else:
        name = "unknown"
	print name
        knockknock_mode(0)
    return name
