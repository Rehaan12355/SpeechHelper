import cv2 as cv
import numpy as np
import mediapipe as mp
import os

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("cannot open camera")
    exit()
    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


mp_face_mesh = mp.solutions.face_mesh

os.makedirs("mouth_frames", exist_ok=True)

frame_count = 0

recording = False

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409
]


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
    
        ret, frame = cap.read()
        if not ret:
            print("Cannot recieve frame, exiting")
            break            
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                mouth_points = []
                
                for index in LIP_LANDMARKS:
                    landmark = face_landmarks.landmark[index]
                    
                    x = int(landmark.x*w)
                    y = int(landmark.y*h)
                    
                    mouth_points.append((x,y))
                
                x_values = [point[0] for point in mouth_points]
                y_values = [point[1] for point in mouth_points]
                
                x_min = min(x_values)
                x_max = max(x_values)

                y_min = min(y_values)
                y_max = max(y_values)

                padding = 20

                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)

                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                
                mouth_crop = frame[y_min:y_max, x_min:x_max]
                if mouth_crop.size > 0:
                    cv.imshow("Mouth", mouth_crop)
                if mouth_crop.size > 0 and recording:
                    filename = f"mouth_frames/frame_{frame_count:05d}.jpg"
                    cv.imwrite(filename, mouth_crop)
                    frame_count += 1
                
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
