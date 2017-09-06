# simplification of the image - thresholding
# everything is either 0 or 1
# some exceptions exist, continue having more values ^ is basic 
import cv2
import numpy as np

# book page has curve and it is dark

img = cv2.imread('jurassic_world.jpg')

retrival, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
# retval form of threshold
# dark picture =  everything greater than 12 is one, and everyhing is below black
# if it is extremely light, use 220

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retrival2, threshold2 = cv2.threshold(grayscale, 12, 255, cv2.THRESH_BINARY)
gauss = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)


cv2.imshow('originial', img)
cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold2)
cv2.imshow('gauss', gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()

# gaussian 

