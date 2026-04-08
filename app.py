import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the box below and click Predict.")

# -----------------------------
# Canvas for drawing
# -----------------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",   # fill color
    stroke_width=15,                       # brush thickness
    stroke_color="white",                  # drawing color
    background_color="black",              # canvas background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert RGBA image to numpy array
        img = canvas_result.image_data

        # Convert to uint8
        img = img.astype("uint8")

        # Convert RGBA to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Resize to 28x28 (MNIST size)
        resized = cv2.resize(gray, (28, 28))

        # Normalize
        processed = resized.astype("float32") / 255.0

        # Reshape for CNN: (1, 28, 28, 1)
        processed = processed.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(processed)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Show processed image
        st.subheader("Processed Image (28x28)")
        st.image(resized, width=150, clamp=True)

        # Show result
        st.success(f"Predicted Digit: {predicted_digit}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please draw a digit first.")