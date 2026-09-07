import cv2 as cv
import numpy as np
import mediapipe as mp
import os
from webcam import open_camera
from landmarks import create_face_mesh, get_mouth_points
from mouth_crop import crop_mouth
from recorder import create_output_folder, save_frame

cap = open_camera()

create_output_folder()


frame_count = 0
recording = False


mp_face_mesh = mp.solutions.face_mesh

os.makedirs("mouth_frames", exist_ok=True)

frame_count = 0

recording = False


with create_face_mesh() as face_mesh:
    while True:
    
        ret, frame = cap.read()
        if not ret:
            print("Cannot recieve frame, exiting")
            break            
        
        mouth_points = get_mouth_points(frame, face_mesh)
        
        if mouth_points:
            mouth_crop = crop_mouth(frame, mouth_points)
            if mouth_crop.size > 0:
                cv.imshow("Mouth", mouth_crop)
            if mouth_crop.size > 0 and recording:
                frame_count = save_frame(mouth_crop, frame_count)
                
        cv.imshow('frame', frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("r"):
            recording = True
            print("Recording")

        elif key == ord("s"):
            recording = False
            print("Stopped")

        elif key == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
