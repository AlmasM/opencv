# basic image operations
import numpy as np
import cv2

img = cv2.imread('jurassic_world.jpg', cv2.IMREAD_COLOR)

px = img[55, 55]

# color value for the pixel
print(px)

# modify pixel to white
img[55, 55] = [255, 255, 255]

print(px)

# ROI - region of image (sub image of image)

# region of the image
roi = img[100:150, 100:150]
# from pixel 100 to pixel 150

print roi

img[100:150, 100:150] = [255, 255, 255]

jur = img[37:111, 107:194]
# 111 - 37 = 74 pixels
# 194 - 107 = 87

# redefine image
img[0:74, 0:87] = jur
 
cv2.imshow('jur', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
