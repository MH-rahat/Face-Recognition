# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:54:39 2020

@author: Tech Land
"""

from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
model = load_model('models/brothers.h5')
image_path_base='dataset3/test_set/rahat/'
output=[]
probabilities=[]
for i in range(1,20):
    image_path=str(image_path_base+str(i)+'.jpg')
    print(image_path)
    
    img = cv2.imread(image_path,3)
    #x = img_to_array(img)
    #x = x/255
    img = cv2.resize(img, (64, 64))
    #img= array_to_img(x)
    img = np.reshape(img, [1, 64, 64, 3])
  
    output.append(model.predict(img))
    probabilities.append(model.predict_proba(img))
    