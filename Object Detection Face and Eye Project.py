# Importing The Required Libraries
import cv2
import numpy as np

# Loading The cascade files after downloading the required haarcascades from
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

video = cv2.VideoCapture(0)
# This will return the video from the first webcam

while True:
    ret, frame = video.read()
    # ret is a boolean which determines whether a Frame has been returned or not.
    # frame stores a frame from the video

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Converting the frame to GrayScale
    # OpenCV reads colors as BGR (Blue, Green, Red) whereas other computer applications read RGB.

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # This code is used to detect the faces in the grayscale image
    # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale

    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(upper left coordinates),(lower right coordinates),(Color of Rectangle),(Thickness))
        # Drawing the rectangle around face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Now to detect eyes
        # we have to find them inside the face
        # so we'll get the region in image in which face is located
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        # This is used to detect the eyes in the grayscale image

        for (ex,ey,ew,eh) in eyes:
            # Drawing the rectangle around the eyes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('OUTPUT',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #This statement just runs once per frame.
    #Basically, if we get a key, and that key is a q, we will exit the while loop with a break, which then runs:

video.release()
cv2.destroyAllWindows()
# This releases the webcam, then closes all of the imshow() windows.
