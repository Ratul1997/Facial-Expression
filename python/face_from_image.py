#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:31:30 2019

@author: rat
"""

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')
mouth_casecade = cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')
eyes_casecade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')

img = cv2.imread('images/Rukon/10.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]

#     eyes = eyes_casecade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#     	print(ex,ey,ew,eh)
#     	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#     	print(roi_color)
    

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()