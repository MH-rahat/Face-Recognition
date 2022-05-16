# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:31:37 2020

@author: Tech Land
"""


import numpy as np
import cv2
import os
import pandas as pd

IMG_DIR = 'D:/Projects/face recognition/dataset4_gray/training_set/riyadh'

for img in os.listdir(IMG_DIR):
        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)

        img_array = (img_array.flatten())

        img_array  = img_array.reshape(-1, 1).T

        print(img_array)

        with open('out3.csv', 'ab') as f:

            np.savetxt(f, img_array, delimiter=",")
            

df = pd.read_csv("out3.csv")
df['pixels'] = df[df.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
df['person']=3
df['usage']='training'
df.to_csv("csvfiles/riyadh_train.csv")
nf = pd.read_csv("csvfiles/riyadh_train.csv")





