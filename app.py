import streamlit as st
import cv2
import numpy as np
from PIL import Image

from core.enhancement import apply_histogram_equalization, apply_gaussian_blur
from core.feature_extract import apply_canny_edge, detect_corners
from core.pattern_detect import detect_faces


st.set_page_config(page_title="IntelliVision", layout="wide")

st.title("IntelliVision: Smart CV Analyzer")
st.write("Upload an image to apply Computer Vision algorithms.")


uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    
    if len(img_array.shape) == 3:
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv2 = img_array

    
    st.sidebar.header("Processing Modules")
    module = st.sidebar.selectbox("Choose Module", ["Enhancement", "Feature Extraction", "Pattern Detection"])
    
    res = None
    action = ""


    if module == "Enhancement":
        action = st.sidebar.radio("Action", ["Histogram Equalization", "Gaussian Blur"])
        if action == "Histogram Equalization":
            res = apply_histogram_equalization(img_cv2)
        else:
            res = apply_gaussian_blur(img_cv2)
            
    elif module == "Feature Extraction":
        action = st.sidebar.radio("Action", ["Canny Edge", "Harris Corners"])
        if action == "Canny Edge":
            res = apply_canny_edge(img_cv2)
        else:
            res = detect_corners(img_cv2)
            
    elif module == "Pattern Detection":
        action = st.sidebar.radio("Action", ["Face Detection"])
        if action == "Face Detection":
            res = detect_faces(img_cv2)
            

    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original")
        st.image(image, use_container_width=True)
        
    with col2:
        st.header(f"Result: {action}")
        if res is not None:
            if len(res.shape) == 2: 
                st.image(res, use_container_width=True, channels="GRAY")
            else:
                res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, use_container_width=True)