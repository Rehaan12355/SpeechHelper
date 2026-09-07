import os
import cv2 as cv

def create_output_folder(folder="mouth_frames"):
    os.makedirs(folder, exist_ok=True)

def save_frame(frame, frame_count, folder="mouth_frames"):
    
    filename = f"{folder}/frame_{frame_count:05d}.jpg"

    cv.imwrite(filename, frame)

    return frame_count + 1    