# Lip Reading Application

A computer vision and machine learning project that recognizes spoken words
from mouth movements without relying on audio.

## Motivation

I started this project to explore how computer vision and machine learning
could make communication more accessible for people who have difficulty
speaking verbally.

The goal is to build a system that captures mouth movements from video,
processes them into a dataset, and eventually trains a model to recognize
words from visual speech.

* Essentially creating a digital lip reader.

## Current Progress

- [x] Webcam video capture with OpenCV
- [x] Face detection
- [x] Facial landmark detection with MediaPipe
- [x] Mouth region extraction
- [x] Mouth frame recording
- [x] Project modularization
- [x] Frame preprocessing and normalization
- [ ] Build and label training dataset
- [ ] Train baseline model
- [ ] Evaluate model performance
- [ ] Real-time word prediction
- [ ] User interface

## How It Works

### 1. Video Capture
OpenCV captures video frames from the webcam.

### 2. Face Landmark Detection
MediaPipe detects facial landmarks in each frame.

### 3. Mouth Extraction
Lip landmarks are used to calculate the mouth region and crop it from
the original frame.

### 4. Preprocessing
Mouth frames are resized and normalized so that samples have a consistent
format for machine learning.

### 5. Dataset Creation
Processed sequences of mouth frames are stored with their corresponding
word labels.

### 6. Model Training
A machine learning model will eventually learn the relationship between
sequences of mouth movements and spoken words.

## Project Structure

    project/
    ├── src/
    │   ├── camera.py
    │   ├── face_detection.py
    │   ├── mouth_crop.py
    │   └── preprocessing.py
    ├── data/
    ├── models/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Machine Learning / Deep Learning

## Running the Project

Clone the repository:

    git clone <repository-url>

Install dependencies:

    pip install -r requirements.txt

Run the application:

    python <main-file>.py

## Development Roadmap

The project is being developed incrementally:

1. Capture video
2. Detect facial landmarks
3. Extract mouth regions
4. Build a labeled dataset
5. Train a baseline visual speech recognition model
6. Evaluate predictions
7. Add real-time inference
8. Build an accessible user interface

## Challenges and Learning

This project is also being used to learn the complete machine learning
development process, including computer vision, dataset construction,
preprocessing, model training, evaluation, and software organization.

## Limitations

Visual speech recognition is inherently ambiguous because different sounds
can produce similar mouth movements. Performance can also be affected by
lighting, camera angle, facial orientation, speaking style, and dataset size.

The initial version of this project is intended as an experimental prototype
rather than a production accessibility system.