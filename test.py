import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)


if not cap.isOpened():
    print("cannot open camera")
    exit()
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot recieve frame, exiting")
        break
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()