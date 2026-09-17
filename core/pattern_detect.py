import cv2
import urllib.request
import os

def detect_faces(image):
    cascade_path = "haarcascade_frontalface_default.xml"
    
    
    if not os.path.exists(cascade_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)
        
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    img_out = image.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_out, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return img_out