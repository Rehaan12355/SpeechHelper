import cv2 as cv
import numpy as np
import mediapipe as mp
import os
from webcam import open_camera
from landmarks import create_face_mesh, get_mouth_points
from mouth_crop import crop_mouth
from recorder import save_frame, create_clip_folder

cap = open_camera()



frame_count = 0

CLIP_LENGTH = 45
recording = False
clip_count = 101
word = input("Enter word to record: ").strip().lower()
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
            
            #Display Mouth
            if mouth_crop.size > 0:
                cv.imshow("Mouth", mouth_crop)
            
            #Recording Block
            if mouth_crop.size > 0 and recording:
                frame_count = save_frame(mouth_crop, frame_count, clip_folder)
                if frame_count >= CLIP_LENGTH:
                    recording = False
                    clip_count += 1                   
        
        
        cv.imshow('frame', frame)
        key = cv.waitKey(1) & 0xFF
        
        
        # Key Controls 
        if key == ord("r"):
            recording = True
            clip_folder = create_clip_folder(word, clip_count)
            frame_count =0
            print("Recording")
            

        elif key == ord("s"):
            recording = False
            print("Stopped")
            clip_count+=1

        elif key == ord("q"):
            break

cap.release()
cv.destroyAllWindows()
