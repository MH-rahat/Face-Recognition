# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:52:53 2020

@author: Tech Land
"""


import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FaceCropper(object):
    CASCADE_PATH = "E:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

    def __init__(self):
            self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result,number,data):
            
            img = cv2.imread(image_path)
            if (img is None):
                print("Can't open image file")
                return 0
    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data=data.append(img)
 
            path = 'dataset4_gray/test_set/forhad'
            cv2.imwrite(os.path.join(path , "%d.jpg" % number), img)
            #cv2.imwrite("image%d.jpg" % i, lastimg)


if __name__ == '__main__':
    image_path_base='dataset3/test_set/forhad/'
    number=0
    data=[]
    for i in range(0,99):
        image_path=str(image_path_base+str(i)+'.jpg')
        isFile = os.path.isfile(image_path)
        if isFile==True:
            print(image_path)
           # image_path='rahat0.png'
            show_result=0
            detecter = FaceCropper()
            detecter.generate(image_path, show_result,number,data)
            number+=1
        else:
            continue
    data=np.reshape(data,[-1,64,64,1])
