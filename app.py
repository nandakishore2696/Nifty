import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras import Sequential
import streamlit as st
from sklearn.neighbors import NearestNeighbors

feature_list = np.load('features_list.npy')
filenames = np.load('filenamesst.npy')

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])

uploaded_file  = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224,3))
    img = img.resize((224, 224)) # Resize image
    image_array = np.asarray(img)  # Convert to NumPy array

    # Display Image (Optional)
    st.image(image_array, caption='Uploaded Image')
    expanded_img_array = np.expand_dims(image_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    preds = model.predict(preprocessed_img)
    preds = preds.flatten()
    preds = preds/norm(preds)

    neighnours = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine')
    neighnours.fit(feature_list)

    distances,indices = neighnours.kneighbors([preds])

    for file in indices[0]:
        temp_img= image.load_img(filenames[file])
        st.image(temp_img)
        
