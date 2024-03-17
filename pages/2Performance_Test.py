#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image

# Define the Streamlit app
def app():


    st.subheader('Testing the Performance of the CNN Classification Model')
    text = """We test our trained model by presenting it with a classification task."""
    st.write(text)
    
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        present_image(uploaded_file)

def present_image(imagefile):
    classifier = st.session_state.classifier
    training_set = st.session_state.training_set
    st.image(imagefile, caption='Smile or No Smile')
    test_image = image.load_img(imagefile, target_size=(32, 32), color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    training_set.class_indices

    if result[0][0]==0:
        prediction = 'No Smile'
    else:
        prediction = 'Smile'

    st.subheader('CNN says the image is class ' + prediction)
 



#run the app
if __name__ == "__main__":
    app()
