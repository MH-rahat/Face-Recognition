import cv2
import sys
import os
import numpy as np
    

class FaceCropper(object):
    CASCADE_PATH = "E:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, img, show_result,number):
            
            #img = cv2.imread(image_path)
            if (img is None):
                print("Can't open image file")
                return 0
    
                lastimg = cv2.resize(img, (256, 256))
                cv2.imshow('img',lastimg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #path ='dataset3/test_set/rahat'
                #path='dataset3/training_set/rahat'
                path='images'
                cv2.imwrite(os.path.join(path , "%d.jpg" % number), lastimg)
                #cv2.imwrite("image%d.jpg" % i, lastimg)


if __name__ == '__main__':
    
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test") 
    
    img_counter = 0
    detecter = FaceCropper()
    number=399
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:                                                                                                                                                                          
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            
            number+=1
            show_result=0
            detecter.generate(frame, show_result,number)
            
        
            
    cam.release()
    
    cv2.destroyAllWindows()
           
