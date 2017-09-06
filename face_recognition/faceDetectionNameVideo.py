import cv2
import numpy as np
import os
import math
import sys
from matplotlib import pyplot as plt
from IPython.display import clear_output


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



def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        if(person != '.DS_Store'):
            labels_dic[i] = person
            # print person
            for image in os.listdir("people/" + person):
                images.append(cv2.imread("people/" + person + '/' + image, 0))
                labels.append(i)
    return (images, np.array(labels), labels_dic)



images, labels, labels_dic = collect_dataset()


# rec_eig = cv2.face.createEigenFaceRecognizer()
# rec_eig.train(images, labels)

rec = cv2.face.createLBPHFaceRecognizer()
rec.train(images, labels)

# rec = cv2.face.createFisherFaceRecognizer()
# rec.train(images, labels)


print 'Model Training Complete'

webcam = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('frontal_face.xml')
threshold = 105

while True:
    ret, img = webcam.read()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_coord = face_cascade.detectMultiScale(frame, 1.2, 5)
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            # collector = cv2.face.MinDistancePredictCollector()
            # rec_lbph.predict(face, collector)
            # conf = collector.getDist()
            # pred = collector.getLabel()
            if sys.version == "3.1.0":
                collector = cv2.face.MinDistancePredictCollector()
                rec.predict(face, collector)
                conf = collector.getDist()
                pred = collector.getLabel()
            else:
                pred, conf = rec.predict(face)
                print "Prediction: " + str(pred)
                print 'Confidence: ' + str(round(conf))
                print 'Threshold: ' + str(threshold)


            print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
            clear_output(wait = True)
            if conf > threshold: # apply threshold
                cv2.putText(img, labels_dic[pred].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            else:
                cv2.putText(img, "Unknown",
                            (faces_coord[i][0], faces_coord[i][1]),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
        draw_rectangle(img, faces_coord) # rectangle around face
    cv2.putText(img, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("Face Camera", img) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break