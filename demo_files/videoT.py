# capturing video
# having the camera show two instance: gray and normal
# save and output the recorded video 

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#  outputting the file

# capturing the video
#  cap1 = cv2.VideoCapture('output.mp4')

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))


while True:
	ret, frame = cap.read()
# convert color
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	out.write(frame)
	cv2.imshow('frame', frame)
	cv2.imshow('gray', gray)
	
	if cv2.waitKey(2) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows() 