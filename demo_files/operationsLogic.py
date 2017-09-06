# image arithmetic and operations - 5

import numpy as np
import cv2

img1 = cv2.imread('./obama.jpg')
img2 = cv2.imread('./biden.jpg')
# both images are identical

# addition operation

add = img1 + img2
#  => (155, 211, 79) + (50, 170, 200) = 205, 381, 279... translated to 205, 255, 255
# cv2.imshow('adding two', add)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# build in add operation

add = cv2.add(img2, img1)
# cv2.imshow('built in add', add)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
# image1, weight of image 1, image2, weight of image2, gamma
# cv2.imshow('addWeighted', weighted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# region of image
# logical operations

img2 = cv2.imread('sentdex.png')

# threshold 
# put logo in the top right corner

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# apply threshold

ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
# if pixel value above 220, it will be converted to 255 white; 
# if it is below 220, will be converted to black
# flipping because inverse

mask_inv = cv2.bitwise_not(mask)
# low level logical operation 
# bitwise_or 
# bitwise_xor (exclusive or - only one of must be true, if true true => false)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# take the chunk from the roi (i.e background picture roi) 			
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.imshow('img2gray', img2gray)
cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img2_fg', img2_fg)
cv2.imshow('dst', dst)




# cv2.imshow('mask', mask)


cv2.waitKey(0)
cv2.destroyAllWindows()
