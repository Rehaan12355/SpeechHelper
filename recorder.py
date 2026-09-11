import os
import cv2 as cv



def save_frame(frame, frame_count, folder):
    
    filename = f"{folder}/frame_{frame_count:05d}.jpg"

    cv.imwrite(filename, frame)

    return frame_count + 1    

def create_clip_folder(word, clip_count):
    
    clip_folder = f"data/{word}/clip_{clip_count:03d}"
    
    print(clip_folder)
    os.makedirs(clip_folder, exist_ok=True)
    
    return clip_folder