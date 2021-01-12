import cv2,glob

gimg = glob.glob("*.jpg")
face = cv2.CascadeClassifier('C:/Users/name/Desktop/python_codes/opencv/cascade_files/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('C:/Users/name/Desktop/python_codes/opencv/cascade_files/haarcascade_eye.xml')

for img in gimg:
        img = cv2.imread(img)   
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #cv2.imshow("gray",cv2.cvtColor(img,cv2.COLOR_BGR2HSV))
        faces = face.detectMultiScale(gray,1.27,5)
        for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                egray = gray[y:y+h , x:x+w]
                eframe = img[y:y+h , x:x+w]
                eyes = eye.detectMultiScale(egray,1.28,5)
                for (ex,ey,ew,eh) in eyes:
                	cv2.rectangle(eframe,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(100)
