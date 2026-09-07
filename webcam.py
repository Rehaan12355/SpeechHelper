import cv2 as cv

def open_camera(camera_index=0):
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("cannot open camera")
        exit()
    return cap