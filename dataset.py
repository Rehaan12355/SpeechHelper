import os
import numpy as np
from preprocess import preprocess_clip
from sklearn.model_selection import train_test_split
LABELS = {
    'hello': 0,
    'yes': 1
}

def load_dataset(data_dir):
    X = []
    y = []
    
    for word in sorted(os.listdir(data_dir)):
        word_path = os.path.join(data_dir, word)
        
        if not os.path.isdir(word_path):
            continue
        if word not in LABELS:
            continue
        for clip in sorted(os.listdir(word_path)):
            clip_path = os.path.join(word_path, clip)
            
            if not os.path.isdir(clip_path):
                continue
            
            processed_clip = preprocess_clip(clip_path)
            X.append(processed_clip)
            y.append(LABELS[word])
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def split_dataset(X, y):
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test 
    
        
        
if __name__ == "__main__":
    
    X, y = load_dataset("data")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    
    
    
    
    
    
        