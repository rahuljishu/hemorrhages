import streamlit as st
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, disk
from skimage import exposure, morphology, filters
from PIL import Image

# Method 1: Vessel leakage detection using Frangi filter
@st.cache_data
def method1_vessel_leakage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vessel_enhanced = frangi(gray)
    _, vessel_binary = cv2.threshold((vessel_enhanced * 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    vessel_binary = remove_small_objects(vessel_binary.astype(bool), min_size=100).astype(np.uint8) * 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_areas = cv2.bitwise_or(red_mask1, red_mask2)
    damaged_areas = cv2.bitwise_and(vessel_binary, red_areas)
    highlighted = img.copy()
    highlighted[damaged_areas == 255] = [0, 0, 255]
    return highlighted, red_areas

# Method 2: Hemorrhage detection pipeline
@st.cache_data
def method2_hemorrhage_detection(img):
    resized = cv2.resize(img, (512, 512))
    green = resized[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_green = clahe.apply(green)
    complement = 255 - clahe_green
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(complement, cv2.MORPH_OPEN, kernel)
    subtracted = cv2.subtract(complement, opened)
    median = cv2.medianBlur(subtracted, 5)
    final_sub = cv2.subtract(median, opened)
    adjusted = exposure.rescale_intensity(final_sub, in_range=(50, 200))
    final_complement = 255 - adjusted
    thresh = filters.threshold_local(final_complement, block_size=51, offset=10)
    binary = final_complement > thresh
    closed = morphology.binary_closing(binary, footprint=disk(3))
    return resized, clahe_green, complement, adjusted, (binary.astype(np.uint8) * 255), (closed.astype(np.uint8) * 255)

# Streamlit UI
st.title("Fundus Image Vessel Leakage & Hemorrhage Detection")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Failed to load image. Please upload a valid image file.")
    else:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

        tab1, tab2 = st.tabs(["Vessel Leakage Detection", "Hemorrhage Detection"])
        with tab1:
            highlighted, red_areas = method1_vessel_leakage(img)
            st.subheader("Detected Hemorrhage Areas")
            st.image(red_areas, caption="Hemorrhage Areas", use_column_width=True)
            st.subheader("Blood Vessel Leakage Highlighted")
            st.image(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB), use_column_width=True)

        with tab2:
            resized, clahe_green, complement, adjusted, binary, closed = method2_hemorrhage_detection(img)
            st.subheader("CLAHE Green Channel")
            st.image(clahe_green, caption="CLAHE Green", use_column_width=True)
            st.subheader("Complemented CLAHE")
            st.image(complement, caption="Complement", use_column_width=True)
            st.subheader("Adjusted Intensity")
            st.image(adjusted, caption="Adjusted", use_column_width=True)
            st.subheader("Region Growing (Binary)")
            st.image(binary, caption="Binary", use_column_width=True)
            st.subheader("Final Hemorrhage Detection")
            st.image(closed, caption="Closed Binary", use_column_width=True)
