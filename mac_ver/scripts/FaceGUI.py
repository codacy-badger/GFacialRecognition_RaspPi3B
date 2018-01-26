#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auther: Guo Yang <guoyang@webmail.hzau.edu.cn>

This file is part of <<ImgProcessing A>> Project.

Summary: GUI program for project

"""

import tkMessageBox
from Tkinter import *
import os

user_path = '/Users/simonguo/myDocuments/Final_project/mac_ver/scripts/' # In order to run, one need to change this user_path variable to the absolute path in your own device.
user_python = 'python2.7' # If your computer has multiple versions of python along with Python 2.7.13, set user_python to 'python2.7' is the most secure way. If not, then change user_python variable to 'python'
def button_test():
    pwd = name_to_be_added.get()
    tkMessageBox.showinfo(title='aaa', message=pwd)


def add_person():
    #print name_to_be_added.get()
    cmd_to_be_run = user_python + ' ' + user_path + 'AddDataSet.py ' + name_to_be_added.get()
    #print cmd_to_be_run
    photo_control_gui(cmd_to_be_run)


def on_click_admin():
    if pwd.get() == "1111" and name_to_be_added.get() != "":
        #tkMessageBox.showinfo(title='Pass', message='ok')
        root1 = Tk()
        root1.title('Register new faces')
        root1.geometry('400x400')
        Label(root1, text='ready for photo shoot?', font=('Arial', 20)).pack()
        Button(root1, width=20, text="ok", command=add_person).pack()
        root1.mainloop()
    else:
        tkMessageBox.showinfo(title='Warning', message='operation cannot be performed')


def on_click_run_cmd(cmd_to_be_run):
    os.system(cmd_to_be_run)


def photo_control_gui(cmd_to_be_run):
    root2 = Tk()
    root2.title('photo control')
    root2.geometry('300x200')
    Label(root2, height=5, text=' ').pack()
    Button(root2, width=20, text="open camera", command=lambda: on_click_run_cmd(cmd_to_be_run)).pack()
    Label(root2, text='press \'p\' to take photo\n press \'esc\' or \'q\' to stop', font=('Arial', 20)).pack()
    root2.mainloop()


def on_ckick_train():
    cmd_for_train = user_python + ' ' + user_path + 'Dataset_train.py'
    os.system(cmd_for_train)

def on_ckick_knockknock():
    cmd_for_train = user_python + ' ' + user_path + 'real_time.py'
    os.system(cmd_for_train)

root = Tk()
root.title('Author: gy Version: 0.0.1')
root.geometry('800x600')

Label(root, text='Facial Recognition implementation using SVM and PCA on a Rasp Pi chip', font=('Arial', 20)).pack()
Label(root, text=' ', font=('Arial', 20)).pack()  # place holder

name_to_be_added = StringVar()   # for string variable used in windows
pwd = StringVar()  # for string variable used in windows

#Label(root, text='', font=('Arial', 20)).pack()
Label(root, text="enter the name of the new face").pack()
root_entry1 = Entry(root, textvariable=name_to_be_added)
root_entry1.pack()

Label(root, text="enter admin password to register a new face").pack()
root_entry2 = Entry(root, textvariable=pwd)
root_entry2['show'] = '*'
root_entry2.pack()

root_Button1 = Button(root, width=20, text="Start registering new face", command=on_click_admin).pack()


# create 2 button module
root_Button2 = Button(root, width=15, text="Train", command=on_ckick_train)
root_Button3 = Button(root, width=15, text="Knock Knock", command=on_ckick_knockknock)

Label(root, text=" ").pack()
Label(root, text=" ").pack()

Label(root, text="To train the dataset, click \'Train\' button below    ").pack()
root_Button2.pack()

Label(root, text=" ").pack()
Label(root, text=" ").pack()

Label(root, text="For REAL_TIME recognition, click \'Knock Knock\' button below    ").pack()
root_Button3.pack()
Label(root, text="press 'q' or 'ESC' to quit Knock Knock mode").pack()
Label(root, text=" ").pack()
Label(root, text=" ").pack()
Label(root, text=" ").pack()
Label(root, text=" ").pack()
Label(root, text="Author: guoyang \nThis GUI program is part of <<ImgProcessing A>> Project.").pack()
# root_Button3.bind("Button-1", on_ckick_knockknock())

root.mainloop()    # entering into message loop
