from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.datasets import mnist
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import cv2
import queue


mnist_model = load_model("/Users/mertaydayanc/Desktop/imageCalculatorVenv/results/operator_model.h5")

originalImage = cv2.imread("/Users/mertaydayanc/Desktop/imageCalculatorVenv/five.png",0)
anotherOne = cv2.imread("/Users/mertaydayanc/Desktop/imageCalculatorVenv/trialImage.png",0)


images =[]

originalImage = imutils.resize(originalImage, height = 400, width = 400)
anotherOne = imutils.resize(anotherOne, height = 400, width = 400)

cv2.imshow('Original Image', originalImage)
cv2.waitKey(0)

cv2.imshow('Side Image', anotherOne)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(originalImage,200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(originalImage,200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

cv2.imshow('second', thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,4), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI


    roi = originalImage[y-30:y+h+30, x-15:x+w+15]


    if w > 15 and h > 5:
        cv2.imwrite('C:\\Users\\Link\\Desktop\\output\\{}.png'.format(i), roi)
        images.append(roi)


#binary
expression = []
for image1 in images:

   height, width = image1.shape[:2]
   if(height > 50 and width > 50):
      cv2.imshow('img', image1)
      ret,thresh = cv2.threshold(image1,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
      kernel = np.ones((5,5), np.uint8)
      img_dilation = cv2.dilate(thresh, kernel, iterations=1)
      img = cv2.resize(img_dilation, (28,28))
      cv2.imshow('img', img)
      cv2.waitKey(0)
      img = img.astype('float32')
      img /= 255
      img = img.reshape(1,784)
      prediction = mnist_model.predict_classes(img)
      print(prediction)
      expression.append(prediction)
      cv2.waitKey(0)
   else:
      print("image Neglected")

print("Expression")
print(expression)
print(eval('4+2'))
str_expression = ""

for a in expression:
   if a[0] == 0:
      str_expression += "0"
   elif a[0] == 1:
      str_expression += "1"
   elif a[0] == 2:
      str_expression += "2"
   elif a[0] == 3:
      str_expression += "3"
   elif a[0] == 4:
      str_expression += "4"
   elif a[0] == 5:
      str_expression += "5"
   elif a[0] == 6:
      str_expression += "6"
   elif a[0] == 7:
      str_expression += "7"
   elif a[0] == 8:
      str_expression += "8"
   elif a[0] == 9:
      str_expression += "9"
   elif a[0] == 10:
      str_expression += "+"
   elif a[0] == 11:
      str_expression += "-"

print('result')
print(str_expression)
print(eval(str_expression))

result = eval(str_expression)

kernel2 = np.ones((10,90), np.uint8)
img_dilation1 = cv2.dilate(thresh2, kernel2, iterations=1)
cv2.imshow('dilatedImage', img_dilation1)

#find contours
contours, hier = cv2.findContours(img_dilation1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

resultStr = str_expression + ' = ' + str(result)
for c in contours:
    x1, y1, w1, h1 = cv2.boundingRect(c)
    if(h1 > 20 and w1 > 50):
       rect = cv2.boundingRect(c)
       x,y,w,h = rect
       cv2.rectangle(originalImage,(x,y),(x+w,y+h),(0,255,0),2)
       cv2.putText(originalImage,resultStr,(x+w-120,y+h +20),0,0.6,(0,255,0))

cv2.imshow("Show",originalImage)

cv2.waitKey(0)
