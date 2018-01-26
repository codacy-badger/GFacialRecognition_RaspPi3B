#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auther: Guo Yang <guoyang@hzau.edu.cn>

This file is part of ImgProcessing A Project.

Summary: LED control on RasPi

"""

import RPi.GPIO as GPIO
import time

GREEN = 13	 # use RasPi's NO.13, 19, 26 GPIO, GND is below GPIO26, on the corner
YELLOW = 19
RED = 26

# Pin Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(GREEN, GPIO.OUT)
GPIO.setup(YELLOW, GPIO.OUT)
GPIO.setup(RED, GPIO.OUT)

def train_mode( ) :
	count = 0
	while(count < 50):
		t = 1
		if t == 1:  # GREEN open
			GPIO.output(RED, False)
			GPIO.output(YELLOW, False)
			GPIO.output(GREEN, True)
			time.sleep(1)
			t = 2
		if t == 2:  # RED open
			GPIO.output(RED, True)
			GPIO.output(YELLOW, False)
			GPIO.output(GREEN, False)
			time.sleep(1)
			t=3
		if t == 3:  # YELLOW open
			GPIO.output(RED, False)
			GPIO.output(YELLOW, True)
			GPIO.output(GREEN, False)
			time.sleep(1)
			t = 1
		count += 1


def knockknock_mode(flag):
	if flag == 1:
		GPIO.output(RED, False)
		GPIO.output(YELLOW, False)
		GPIO.output(GREEN, True)
                time.sleep(1)
	else:
		GPIO.output(RED, True)
		GPIO.output(YELLOW, False)
		GPIO.output(GREEN, False)
                time.sleep(1)

#for i in range(100):
#	if i%2 ==0:	
#		knockknock_mode(1)
#	else:
#		knockknock_mode(0)


