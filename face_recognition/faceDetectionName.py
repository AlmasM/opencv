import numpy
import math
from matplotlib import pyplot as plt
import cv2
import os
from IPython.display import clear_output
# print cv2.__version__ 

# normalize image for learning
face_cascade = cv2.CascadeClassifier('frontal_face.xml')

cv2.startWindowThread()

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    	faces_coord = self.classifier.detectMultiScale(gray, 1.2, 5)

        return faces_coord

class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        # print self.video.isOpened()

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()

def cut_faces(image, faces_coord):
    faces = []
      
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation = cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation = cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (150, 150, 0), 8)



webcam = VideoCamera()
detector = FaceDetector("xml/frontal_face.xml")

cap = cv2.VideoCapture(0)


folder = "people/" + raw_input('Person: ').lower() # input name
if not os.path.exists(folder):
    os.mkdir(folder)
    counter = 1
    timer = 20
    while counter <= 500:
    	ret, img = cap.read()
    	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    	faces_coord = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces_coord):
        	faces = normalize_faces(gray, faces_coord)
        	cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
           	# plt_show(faces[0], "Images Saved:" + str(counter))
        	clear_output(wait = True) # saved face in notebook
        	# plt_show(faces[0])
        	counter += 1;
        draw_rectangle(img, faces_coord)
        cv2.imshow('Picture', img)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
else:
    print "This name already exists."


