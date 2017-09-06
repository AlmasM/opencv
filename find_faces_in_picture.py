# print("Check")
from PIL import Image
import face_recognition
import cv2
# print("Check")



# Load the jpg file into a numpy array
image = face_recognition.load_image_file("friends.jpg")

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location

    # Display the results	

    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "face", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
face_image = image
pil_image = Image.fromarray(face_image)
pil_image.show()