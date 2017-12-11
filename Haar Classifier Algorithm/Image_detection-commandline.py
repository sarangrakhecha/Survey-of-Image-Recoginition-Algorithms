import numpy as np
import cv2
import sys

object_cascade = cv2.CascadeClassifier('/home/user/opencv-3.1.0/opencv-haar-classifier-training/classifier.xml')


args = sys.argv[1]
img = cv2.imread(args)
#print img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('img',img)
objects = object_cascade.detectMultiScale(gray, 1.3, 5)
#cv2.imshow('img',img)
for (x,y,w,h) in objects:
    img = cv2.rectangle(img,(x,y),(x+w-20,y+h-20),(255,0,0),2,)
   

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
