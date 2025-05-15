import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import keras

# Title HTML
html_temp = """
    <div style="background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">X-Ray Image Classifier</h2>
    </div>
"""
print(keras.__version__)
st.markdown(html_temp, unsafe_allow_html=True)

# Constants
img_size = 100
CATEGORIES = ['NORMAL', 'PNEUMONIA']

# Load the model
model = keras.models.load_model('Pneumonia_xray_prediction.h5')
print("Model Loaded")

def load_classifier():
    st.subheader("Upload an X-Ray Image to detect if it is Normal or Pneumonia")
    file = st.file_uploader(label="Upload Image", type=["jpeg", "jpg", "png"])

    if file is not None:
        try:
            # Load and preprocess the image
            img = Image.open(file).convert('RGB')  # Ensure RGB mode
            img = img.resize((img_size, img_size))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize like in Colab
            img_array = np.expand_dims(img_array, axis=0)  # (1, img_size, img_size, 3)

            st.image(img, caption="Uploaded X-ray", use_column_width=True)

            if st.button("PREDICT"):
                prediction = model.predict(img_array)
                confidence = prediction[0][0]
                class_index = int(round(confidence))
                preds = f"**Prediction:** {CATEGORIES[class_index]} \n\n**Confidence:** {confidence:.4f}"
                st.success(preds)

        except Exception as e:
            st.error(f"Error processing image: {e}")

def main():
    load_classifier()

if __name__ == "__main__":
    main()
