# write and draw on the image

import numpy as np
import cv2

img = cv2.imread('jurassic_world.jpg', cv2.IMREAD_COLOR)

# cv2.line(img, (0,0), (150, 150), (255, 255, 255), 15)


# totally blue (255, 0 , 0)
# in opencv it is bgr - blue green red
# normally it is rgb - red green blue
# white 255, 255, 255
# black 0,0,0
# 15 is optional width

# img, where it begins, where it ends, color, width

cv2.rectangle(img, (15,25), (200, 50), (0,255, 0) , 5 )
# image, top left, bottom right, color, optional width(5)

# cv2.circle(img, (100, 63), 55, (0,0,255), -1)
# -1 fills in 
# img, center, radius, color, fill

pts = np.array([[100, 200], [40,20], [90,70], [30,45]], np.int32)
pts = pts.reshape((-1, 1, 2))
# cv2.polylines(img, [pts], True, (0,255, 255), 3)
# true first and last connected or not


# write on picture
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV text!', (0,130), font, 1, (200, 255, 255), 2, 1)
# image, the text itself, starting point, font, size of the font, color, width

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()