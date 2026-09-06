import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot recieve frame, exiting")
        break
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
