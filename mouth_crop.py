def crop_mouth(frame, mouth_points, padding=20):
    x_values = [point[0] for point in mouth_points]
    y_values = [point[1] for point in mouth_points]
    
    x_min = min(x_values)
    x_max = max(x_values)

    y_min = min(y_values)
    y_max = max(y_values)
    
    h,w, _ = frame.shape
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    
    
    mouth_crop = frame[y_min:y_max, x_min:x_max]   
    
    return mouth_crop 