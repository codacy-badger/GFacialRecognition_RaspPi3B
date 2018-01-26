"""
Auther: Guo Yang <guoyang@hzau.edu.cn>

This file is part of ImgProcessing A Project.

Summary: SVM methods using Scikit

"""

import tkMessageBox
import warnings
from time import time

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.cross_validation import train_test_split

from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


def build_SVC(face_profile_data, face_profile_name_index, face_dim):
    """
    Build the SVM classification module using the face_profile_data matrix (numOfFace X numOfPixel) and face_profile_name_index array, face_dim is a tuple of the dimension of each image(h,w) Returns the SVM classification modle
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
        The pca that contains the top 150-200 eigenvectors extracted using approximated Singular Value Decomposition of the data

    """

    X = face_profile_data
    y = face_profile_name_index

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 200 # maximum number of components to keep

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


def errorRate(pred, actual):
    if pred.shape != actual.shape: return None
    error_rate = np.count_nonzero(pred - actual)/float(pred.shape[0])
    return error_rate


def predict_single(clf, pca, img, face_profile_names):
    img = img.ravel()
    principle_components = pca.transform(img.reshape(1, -1))  # +++++++++++++++++++++++
    # print principle_components
    proba = clf.predict_proba(principle_components)  # probability+++++++++++++++++++++++
    pred = clf.predict(principle_components)
    print proba[0][pred]
    if proba[0][pred] > 0.5:
        name = face_profile_names[int(pred)]
    else:
        name = "unknown"
    return name
