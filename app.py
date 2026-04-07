import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (PNG / JPG / JPEG).")
st.info("Recommended: white digit on black background, 28x28 image for best results.")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras", compile=False)

model = load_model()

# ---------------------------
# Preprocess image
# ---------------------------
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(image)

    # Auto invert if background is white
    # If average brightness is high, assume white background
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN: (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# ---------------------------
# Upload file
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Show original image
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess
    processed = preprocess_image(image)

    # Show processed 28x28 image
    st.image(
        processed[0].reshape(28, 28),
        caption="Processed Image (28x28 for CNN)",
        width=200,
        clamp=True
    )

    # Predict
    prediction = model.predict(processed)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Show results
    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.2f}%")