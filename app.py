import io
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import streamlit as st
import threading

# FastAPI Backend
app = FastAPI()

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='app\model_self_op2.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

model_classes = ['lung adenocarcinomas', 'benign lung tissues', 'lung squamous cell carcinomas']

# Function to preprocess image for FastAPI
def preprocess_image(img):
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to predict class and confidence using FastAPI
def predict_image_class(img):
    img = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = model_classes[predicted_class]
    return class_name, confidence

# Streamlit Frontend
st.title("Lung Cancer Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    try:
        pil_image = Image.open(uploaded_file)
        if st.button('Classify'):
            class_name, confidence = predict_image_class(pil_image)
            confidence_percentage = int(confidence * 100)
            st.write(f"Predicted Class: {class_name}")
            st.write(f"Confidence: {confidence_percentage}%")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# Run both FastAPI and Streamlit in separate threads
def run_fastapi():
    uvicorn.run(app, host='0.0.0.0', port=8000)

def run_streamlit():
    st.run()

fastapi_thread = threading.Thread(target=run_fastapi)
streamlit_thread = threading.Thread(target=run_streamlit)

fastapi_thread.start()
streamlit_thread.start()
