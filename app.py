import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ----------------------- DCP Parameter Presets -----------------------
PRESETS = {
    "thin": {"window_size": 15, "omega": 0.75, "t0": 0.05, "percent": 0.0008},
    "moderate": {"window_size": 25, "omega": 0.85, "t0": 0.03, "percent": 0.0015},
    "thick": {"window_size": 35, "omega": 1.0, "t0": 0.01, "percent": 0.0025},
}

# ----------------------- DCP Functions -----------------------

def get_dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(min_channel, kernel)

def estimate_atmospheric_light(image, dark_channel, top_percent):
    num_pixels = int(top_percent * dark_channel.size)
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[-num_pixels:], dark_channel.shape)
    brightest = image[indices]
    return np.max(brightest, axis=0)

def estimate_transmission(image, A, omega, window_size):
    normed = image / A
    dark_channel = get_dark_channel(normed, window_size)
    return 1 - omega * dark_channel

def guided_filter(I, p, radius=40, eps=1e-3):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    var_I = corr_I - mean_I ** 2
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, cv2.CV_64F, (radius, radius)) * I + cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

def recover_image(image, transmission, A, t0):
    transmission = np.clip(transmission, t0, 1)[:, :, np.newaxis]
    J = (image - A) / transmission + A
    return np.clip(J, 0, 1)

def apply_clahe(image_rgb):
    lab = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB) / 255.0

def dehaze_adaptive_dcp(input_bgr, haze_level="thick"):
    params = PRESETS[haze_level]
    image_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB) / 255.0
    dark_channel = get_dark_channel(image_rgb, params["window_size"])
    A = estimate_atmospheric_light(image_rgb, dark_channel, params["percent"])
    transmission = estimate_transmission(image_rgb, A, params["omega"], params["window_size"])
    gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    refined = guided_filter(gray, transmission)
    dehazed = recover_image(image_rgb, refined, A, params["t0"])
    return (apply_clahe(dehazed) * 255).astype(np.uint8)

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Satellite Image Dehazer", layout="centered")
st.title("🛰️ Satellite/Drone Image Dehazing")
st.write("Upload a hazy satellite or drone image and choose the haze level. The app will enhance it using adaptive DCP dehazing.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
haze_level = st.selectbox("☁️ Select haze level", ["thin", "moderate", "thick"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, 1)

    st.image(bgr_img[:, :, ::-1], caption="Original Image", use_container_width=True)

    with st.spinner("⏳ Dehazing in progress..."):
        result_img = dehaze_adaptive_dcp(bgr_img, haze_level)

    st.image(result_img, caption="🧠 Dehazed Image", use_container_width=True)

    st.download_button(
        label="📥 Download Dehazed Image",
        data=cv2.imencode(".jpg", result_img)[1].tobytes(),
        file_name="dehazed_output.jpg",
        mime="image/jpeg"
    )
