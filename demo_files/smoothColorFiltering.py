# smoothing color filter
# 1) copy from colorFiltering
# 2)smoothing and filtering noise


# color filtering
import cv2
import numpy as np

# operation with green screen and chaning it is an example

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()

	# wont use underscore value

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# way to define color
	# hue - is color value
	# value - how much of that color is in existence
	# saturation - intensity of that color
	# each is independent
	# hue, saturation, value
	# rgb are not independent 

	# hsv hue sat value
	lower_red = np.array([150, 150, 50])

	upper_red = np.array([180,255,150])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	# if it is in range, then 1
	res = cv2.bitwise_and(frame, frame, mask = mask)
	# dark_red = np.uint8([[[12,22,121]]])
	# dark_red = cv2.cvtColor(dark_red, cv2.COLOR_BGR2HSV)
	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('res', res)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release()

