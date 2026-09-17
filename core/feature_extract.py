import cv2
import numpy as np

def apply_canny_edge(image, lower=100, upper=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, lower, upper)

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None) 
    
    img_out = image.copy()
    
    img_out[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img_out