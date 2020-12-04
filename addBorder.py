import numpy as np
import cv2

im = cv2.imread('plus.png')
row, col= im.shape[:2]
bottom= im[row-2:row, 0:col]
mean= cv2.mean(bottom)[0]

bordersize=20
border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )

cv2.imshow('image',im)
cv2.imshow('bottom',bottom)
cv2.imshow('border',border)
cv2.waitKey(0)
cv2.destroyAllWindows()
