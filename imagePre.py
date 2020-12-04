from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

images =[]
img = cv2.imread("/Users/MertayDayanc/Desktop/trialImage.JPG")

image = imutils.resize(img, height = 500)
#grayscale 
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
cv2.imshow('gray', gray) 

#binary 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) 
cv2.imshow('second', thresh) 

#dilation 
kernel = np.ones((1,1), np.uint8) 
img_dilation = cv2.dilate(thresh, kernel, iterations=1) 
cv2.imshow('dilated', img_dilation) 

#find contours 
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#sort contours 
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs): 
    # Get bounding box 
    x, y, w, h = cv2.boundingRect(ctr) 
    
    # Getting ROI 
    roi = image[y-15:y+h+15, x-15:x+w+15] 
    # show ROI 
    #cv2.imshow('segment no:'+str(i),roi) 
    #cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 
    #cv2.waitKey(0) 
    if w > 15 and h > 15: 
        cv2.imwrite('C:\\Users\\Link\\Desktop\\output\\{}.png'.format(i), roi)
        images.append(roi)

cv2.waitKey(0)