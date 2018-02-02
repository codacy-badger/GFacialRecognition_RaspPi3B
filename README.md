# FacialRecognition_RaspPi3B

my own Facial Recognition implementation using SVM and PCA on a Rasp Pi 3B chip

## environment

- macOS 10.12.6 + PyCharm
    - python (2.7.13)
    - opencv-python (3.3.0.10)
    - scikit-learn (0.19.1)
    - Tkinter
    - numpy (1.13.3)
    - scipy (1.0.0)
    - numpy (1.13.3)

## For mac version instead of Pi version, please proceed to mac_version branch

[mac_version branch portal](https://github.com/sgyzetrov/GFacialRecognition_RaspPi3B/tree/mac_version)

## snapshots

### GUI:

![GUI.png](http://img.blog.csdn.net/20180126143314221)

As you can see to register a new face require admin password, I set to '1111', and you can change it in `FaceGUI.py`

### hardware:

![pi.png](http://img.blog.csdn.net/20180126140958970)

### Facial Recognition Result

![admin.png](http://img.blog.csdn.net/20180126143531077)

## Usage

In order to run, one need to change the `user_path` variable in `FaceGUI.py` to the absolute path in your own Pi.

Also, If your computer/Pi has multiple versions of Python along with Python 2.7.13, set `user_python` in `FaceGUI.py` to 'python2.7' is the most secure way. If not, then change the `user_python` variable to 'python'

If you finished the above modifications, then, run `FaceGUI.py` and you are all set!

## Welcome to fork and PR

If encounter any troubles, feel free to take them to me in new issues! [click](https://github.com/sgyzetrov/GFacialRecognition_RaspPi3B/issues)
