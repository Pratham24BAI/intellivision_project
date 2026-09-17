import numpy as np
from core.enhancement import apply_gaussian_blur, apply_histogram_equalization
from core.feature_extract import apply_canny_edge
from core.pattern_detect import detect_faces

def test_blur():
    
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    blurred = apply_gaussian_blur(dummy_img, 5)
    
    
    assert blurred.shape == (100, 100, 3)

def test_histogram_equalization():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    eq = apply_histogram_equalization(dummy_img)
    
    
    assert len(eq.shape) == 2

def test_canny_edge():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    edges = apply_canny_edge(dummy_img)
    
    
    assert len(edges.shape) == 2

def test_face_detection():
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detected = detect_faces(dummy_img)
    
    
    assert detected.shape == (100, 100, 3)