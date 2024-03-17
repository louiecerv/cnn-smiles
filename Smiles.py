#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import time

# Define the Streamlit app
def app():
    if "X" not in st.session_state: 
        st.session_state.X = []
    
    if "y" not in st.session_state: 
        st.session_state.y = []

    if "model" not in st.session_state:
        st.session_state.model = []

    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []

    if "X_test_scaled" not in st.session_state:
            st.session_state.X_test_scaled = []

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """Convolutional Neural Network Image Classifier"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('smiles.jpg', caption='The Smiles Dataset')

    text = """
    \nThis Streamlit application demonstrates a Convolutional Neural Network (CNN)
    classifier for image recognition, specifically trained on a dataset of smiles. 
    Users can interact with the app by uploading an image, and the CNN model will 
    predict the presence or absence of a smile in the image.
    \nTech Stack:
    Streamlit: A Python library for creating web apps.
    TensorFlow: Deep learning frameworks for building and training CNNs.
    App Functionality:
    Image Upload: The app provides a user interface for uploading an image file.
    CNN Inference: The image is passed through the trained CNN model, which 
    generates predictions.
    Prediction Display: The app displays the model's prediction on the presence 
    or absence of a smile in the image. 
    Benefits:
    \nUser-friendly Interface: Streamlit simplifies web app development, making 
    the CNN model accessible without coding knowledge.
    Interactive Exploration: Users can experiment with different images 
    and observe the model's performance.
    Educational Tool: The app can be a valuable tool for understanding CNNs and 
    their applications in image classification tasks.
    """
    st.write(text)

    with st.expander("How to use this App"):
         text = """Step 1. Go to Training page. Set the parameters of the CNN. Click the button to begin training.
         \nStep 2.  Go to Performance Testing page and click the button to load the image
         and get the model's output on the classification task.
         \nYou can return to the training page to try other combinations of parameters."""
         st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
