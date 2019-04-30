#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:32:36 2019

@author: rat
"""
import os
import cv2
import numpy as np
from PIL import Image
import pickle
import dlib


y_labels = []
x_train = []
current_id = 0
label_ids = {}

recognizer = cv2.face.FisherFaceRecognizer_create()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            

width_d, height_d = 280, 280 
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                # print(current_id)
            id_ = label_ids[label]
        
            print(path)
            img = cv2.imread(path)
            image_array = cv2.resize(img,(280,280))
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("asas",gray)

            faces = detector(gray)
            data = []
            for (face) in faces:
                for i in range(0,68):
                    landmarks = predictor(gray,face)
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y

                    point = (x,y)
                    data.append(point)


            if not len(data) == 0 :
                x_train.append(np.array(data))
                print(type(x_train), type(data))
                y_labels.append(id_)
        

#print(y_labels)
#print(x_train)
                
with open("labels.pickle","wb") as f:
      pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")
