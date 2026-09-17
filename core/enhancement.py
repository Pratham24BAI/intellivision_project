import cv2

def apply_histogram_equalization(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def apply_gaussian_blur(image, kernel_size=5):
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)