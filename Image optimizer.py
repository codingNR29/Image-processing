import cv2
import numpy as np

def optimize_with_opencv(image_path, output_path):
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    
    # 1. Load the image into a NumPy matrix (Note: OpenCV loads as BGR, not RGB)
    img = cv2.imread(image_path)
    
    # 2. Crop to exactly 9:16 using NumPy array slicing
    h, w = img.shape[:2]
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Image is too wide, slice the x-axis
        new_w = int(target_ratio * h)
        offset = (w - new_w) // 2
        img = img[:, offset:offset+new_w]
    else:
        # Image is too tall, slice the y-axis
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        img = img[offset:offset+new_h, :]
        
    # 3. Downsample using Lanczos interpolation over an 8x8 pixel neighborhood
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
    # 4. Manual Unsharp Masking
    # First, create the low-pass version (blurred)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    
    # Then, mathematically combine them: Sharpened = Original + amount * (Original - Blurred)
    # cv2.addWeighted applies this matrix formula efficiently: alpha*img1 + beta*img2 + gamma
    # Here, alpha is 2.5, beta is -1.5, giving a crisp pop to the edges.
    img = cv2.addWeighted(img, 2.5, blurred, -1.5, 0)
    
    # 5. Save with a controlled JPEG quality threshold
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

# Give it a run
optimize_with_opencv("images/Rubik'sCubes.jpg", "images/optimized_story_Rubik'sCubes.jpg")