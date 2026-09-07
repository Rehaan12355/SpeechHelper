import os


DATA_DIR = "data"
EXPECTED_FRAMES = 45

def validate_clip(clip_path):
    
    files = os.listdir(clip_path)
    
    frame_files = [
        file 
        for file in files 
        if file.endswith(".jpg")
    ]
    
    frame_count = len(frame_files)
    
    return frame_count

def validate_word(word_path):
    
    clips = os.listdir(word_path)
    
    for clip in clips:
        clip_path = os.path.join(word_path, clip)
        
        if not os.path.isdir(clip_path):
            continue
        
        frame_count = validate_clip(clip_path)
        
        if frame_count == EXPECTED_FRAMES:
            print(f"{clip}: {frame_count} frames")      
        else:
            print(f"{clip}: {frame_count} frames <-- problem")
            
def validate_dataset():
    
    words = os.listdir(DATA_DIR)
    
    for word in words:
        word_path = os.path.join(DATA_DIR, word)
        if not os.path.isdir(word_path):
            continue
        
        print(f"Checking word: {word}")
        
        validate_word(word_path)
        
if __name__ == "__main__":
    validate_dataset()