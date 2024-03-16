#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import time

# Define the Streamlit app
def app():
    classifier = st.session_state.classifier
    training_set = st.session_state.training_set
    
    st.subheader('Testing the Performance of the CNN Classification Model')
    text = """We test our trained model by presenting it with a classification task."""
    st.write(text)
    
    if st.button('Begin Test'):
        st.image('dataset/single_prediction/cat_or_dog_1.jpg', caption='Cat or Dog Test Image 1')
        test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)
        training_set.class_indices

        if result[0][0]==0:
            prediction = 'dog'
        else:
            prediction = 'cat'

        st.subheader('CNN model says the image is a ' + prediction)
 



#run the app
if __name__ == "__main__":
    app()
