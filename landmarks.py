import mediapipe as mp
import cv2 as cv

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409
]

def create_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_mouth_points(frame, face_mesh):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        mouth_points = []
        
        for index in LIP_LANDMARKS:
            landmark = face_landmarks.landmark[index]
            
            x = int(landmark.x*w)
            y = int(landmark.y*h)
            
            mouth_points.append((x,y))
        return mouth_points