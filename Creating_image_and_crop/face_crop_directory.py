import cv2
import sys
import os


class FaceCropper(object):
    CASCADE_PATH = "E:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result,number):
            
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (img is None):
                print("Can't open image file")
                return 0
    
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(img, 1.3, 5, minSize=(100, 100))
            if (faces is None):
                print('Failed to detect face')
                return 0
    
            if (show_result):
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
            facecnt = len(faces)
            print("Detected faces: %d" % facecnt)
            i = 0
            height, width = img.shape[:2]
    
            for (x, y, w, h) in faces:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int((r * 2))-10
    
                faceimg = img[ny:ny+nr, nx:nx+nr]
                lastimg = cv2.resize(faceimg, (64, 64))
                i += 1
                path = 'dataset4_gray/test_set/forhad'
                cv2.imwrite(os.path.join(path , "%d.jpg" % number), lastimg)
                #cv2.imwrite("image%d.jpg" % i, lastimg)


if __name__ == '__main__':
    image_path_base='dataset3/training_set/forhad/'
    number=0
    for i in range(0,109):
        image_path=str(image_path_base+str(i)+'.jpg')
        isFile = os.path.isfile(image_path)
        if isFile==True:
            print(image_path)
           # image_path='rahat0.png'
            show_result=0
            detecter = FaceCropper()
            detecter.generate(image_path, show_result,number)
            number+=1
        else:
            continue
