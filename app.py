import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Image Detection App", layout="centered")

st.title("🖼️ Image Detection App")
st.write("Upload an image and get predictions")

# ✅ Cache model
@st.cache_resource
def load_model():
    model = YOLO(r"C:/Users/user\Desktop/tauhid_THERMAL/train_results/elephant_yolov8n_multi_class/weights/best.pt")
    model.to("cpu")   # safer
    return model

try:
    model = load_model()
    st.success("✅ Model loaded")

except Exception as e:
    st.error("❌ Error loading model")
    st.exception(e)
    st.stop()

# ✅ File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)

        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        st.write("🔍 Running inference...")

        # ✅ Convert PIL → numpy
        img_array = np.array(image)

        results = model(img_array)

        result_image = results[0].plot()

        st.image(result_image, caption="✅ Prediction Result", use_container_width=True)

        st.success("🎉 Done")

    except Exception as e:
        st.error("❌ Inference error")
        st.exception(e)

else:
    st.info("👆 Upload an image to start")
