# Lung Cancer Classification Web App

This project demonstrates the development of a web application for lung cancer image classification using a Convolutional Neural Network (CNN) model. The trained model achieves over 90% accuracy on training, validation, and test datasets. The app is built using Streamlit and FastAPI, providing an easy-to-use interface for users to classify lung cancer images in real-time. The application is deployed on Streamlit Share, making it accessible to a wider audience.

## Project Overview

The main steps of the project are as follows:

1. **Data Collection and Preprocessing:**
   - The lung cancer dataset was collected and preprocessed to create training, validation, and test sets.
   - Images were resized, normalized, and augmented to enhance the model's robustness.

2. **Model Training:**
   - A custom CNN architecture was designed and trained on the prepared dataset.
   - The model achieved over 90% accuracy on the training, validation, and test datasets.

3. **Conversion to TensorFlow Lite:**
   - The trained CNN model was converted to TensorFlow Lite format to optimize for deployment on resource-constrained environments.

4. **Web Application Development:**
   - Streamlit was used to create an interactive frontend for users to upload lung cancer images and handle image classification using the TensorFlow Lite model.

5. **Deployment:**
   - The application was deployed on Streamlit Share, providing a public URL for users to access the lung cancer classification tool.

Author Shubham Gupta @ https://github.com/shubham-mg/lungcancer
deployed site url:https://lungcancer1.streamlit.app
