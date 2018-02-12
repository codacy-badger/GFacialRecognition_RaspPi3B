# FacialRecognition_RaspPi3B

[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
[![Python 2.7](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/)

my own Facial Recognition implementation using SVM and PCA on a Rasp Pi 3B chip

This branch is for my code to run in macOS in order to test and development. 

## environment

- macOS 10.12.6 + PyCharm
    - python (2.7.13)
    - opencv-python (3.3.0.10)
    - scikit-learn (0.19.1)
    - Tkinter
    - numpy (1.13.3)
    - scipy (1.0.0)
    - numpy (1.13.3)

## For Pi version instead of mac version, please proceed to master branch

[master branch portal](https://github.com/sgyzetrov/GFacialRecognition_RaspPi3B)

## snapshots

### GUI:

![GUI.png](https://github.com/sgyzetrov/GFacialRecognition_RaspPi3B/blob/master/img/GUI.png)

As you can see to register a new face require admin password, I set to '1111', and you can change it in `FaceGUI.py`

### hardware:

![pi.png](https://github.com/sgyzetrov/GFacialRecognition_RaspPi3B/blob/master/img/pi.png)

### Facial Recognition Result

![admin.png](https://github.com/sgyzetrov/GFacialRecognition_RaspPi3B/blob/master/img/admin.png)

## Usage

In order to run, one need to change the `user_path` variable in `FaceGUI.py` to the absolute path in your own Pi.

Also, If your computer/Pi has multiple versions of Python along with Python 2.7.13, set `user_python` in `FaceGUI.py` to 'python2.7' is the most secure way. If not, then change the `user_python` variable to 'python'

If you finished the above modifications, then, run `FaceGUI.py` and you are all set!

## Welcome to fork and PR

If encounter any troubles, feel free to take them to me in new issues!
