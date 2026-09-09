import cv2 as cv
import os
import numpy as np
FRAME_WIDTH = 128

FRAME_HEIGHT = 64


def preprocess_frame(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    resized_frame = cv.resize(
        gray_frame, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    
    resized_frame = resized_frame.astype('float32') / 255
    resized_frame = np.expand_dims(resized_frame, axis=-1)
    return resized_frame


def preprocess_clip(clip_path):
    frame_files = os.listdir(clip_path)

    frame_files = [
        file
        for file in frame_files
        if file.endswith(".jpg")
    ]
    frame_files.sort()

    processed_frames = []

    for file in frame_files:
        frame_path = os.path.join(clip_path, file)
        frame = cv.imread(frame_path)

        if frame is None:
            continue
        processed_frame = preprocess_frame(frame)
        
        processed_frames.append(processed_frame)
        
    processed_frames = np.array(processed_frames)
    return processed_frames


if __name__ == "__main__":
    clip = preprocess_clip("data/hello/clip_000")
    
    print("Shape:", clip.shape)
    print("Data type", clip.dtype)
    print("Minimum value", clip.min())
    print("Max value", clip.max())

