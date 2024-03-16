#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

# Define the Streamlit app
def app():

    if "classifier" not in st.session_state:
        st.session_state.classifier = []

    if "training_set" not in st.session_state:
        st.session_state.training_set = []
    
    if "test_set" not in st.session_state:
        st.session_state.test_set = []
        
    text = """
    A convolutional neural network (CNN) is a type of artificial intelligence especially
    good at processing images and videos.  Unlike other neural networks, CNNs don't need 
    images to be pre-processed by hand. Instead, they can learn to identify features 
    themselves through a process called convolution.
    Layers: CNNs are built up of layers, including an input layer, convolutional 
    layers, pooling layers, and fully-connected layers.
    Convolutional layers: These layers use filters to identify patterns and features 
    within the image. Imagine a filter like a small magnifying glass that scans the image 
    for specific details.
    Pooling layers: These layers reduce the complexity of the image by summarizing the 
    information from the convolutional layers.
    Fully-connected layers: These layers work similarly to regular neural networks, 
    taking the outputs from the previous layers and using them to classify the image 
    or make predictions."""
    st.write(text)

    progress_bar = st.progress(0, text="Loading the images, please wait...")

    # Initialize the CNN
    classifier = keras.Sequential()

    # Convolutional layer
    classifier.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))  # Add input shape for RGB images

    # Max pooling layer
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten layer
    classifier.add(layers.Flatten())

    # Dense layers
    classifier.add(layers.Dense(units=128, activation="relu"))
    classifier.add(layers.Dense(units=1, activation="sigmoid"))

    # Compile the model
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    st.session_state.classifier = classifier

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, horizontal_flip=True)

    # Data preparation
    training_set = train_datagen.flow_from_directory(
        "dataset/training_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
    )
    test_set = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
    )

    st.session_state.training_set = training_set
    st.session_state.test_set = test_set

    # update the progress bar
    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    # Progress bar reaches 100% after the loop completes
    st.success("Image dataset loading completed!") 

    # Get the data for the first 25 images in training set
    train_data = next(training_set)
    train_images, train_labels = train_data[0][0:25], train_data[1][0:25]  # Get first 25 images and labels

    # Plot the training set images
    plot_images(train_images, train_labels)

   # Define CNN parameters    
    st.sidebar.subheader('Set the CNN Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["adam", "lbfgs", "sgd"]
    optimizer = st.sidebar.selectbox('Select the optimizer:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=250,
        value=10,  # Initial value
    )

    epochs = st.sidebar.slider(   
        label="Set the epochs:",
        min_value=3,
        max_value=30,
        value=3
    )

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the model please wait...")
        # Train the model
        batch_size = 64
        training_set = st.session_state.training_set
        test_set = st.session_state.test_set

        # Train the model
        classifier.fit(
            training_set,
            epochs=epochs,
            validation_data=test_set,
            steps_per_epoch=4,
            validation_steps=10,
            callbacks=[CustomCallback()]
        )
        
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Model training completed!") 
        st.write("Use the sidebar to open the Performance page.")

# Define a function to plot images
def plot_images(images, labels):
    fig, axs = plt.subplots(5, 5, figsize=(10, 6))  # Create a figure with subplots

    # Flatten the axes for easier iteration
    axs = axs.flatten()

    for i, (image, label) in enumerate(zip(images, labels)):
        axs[i].imshow(image)  # Use ax for imshow on each subplot
        axs[i].set_title(f"Class: {label}")  # Use ax.set_title for title
        axs[i].axis("off")  # Use ax.axis for turning off axis

    plt.tight_layout()  # Adjust spacing between subplots
    st.pyplot(fig)

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.write(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

#run the app
if __name__ == "__main__":
    app()
