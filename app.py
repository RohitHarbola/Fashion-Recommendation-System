import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 

st.header('Fashion Recommendation System')

# Load precomputed features and filenames
Image_features = pkl.load(open('Image_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Fit the NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Initialize session state for favorites
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []

# File uploader
upload_file = st.file_uploader("Upload Image")

if upload_file is not None:
    # Ensure the "upload" directory exists
    upload_dir = 'upload'
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file to the "upload" directory
    file_path = os.path.join(upload_dir, upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display the uploaded image
    st.subheader('Uploaded Image')
    st.image(file_path)

    # Extract features and get recommendations
    input_img_features = extract_features_from_images(file_path, model)
    distances, indices = neighbors.kneighbors([input_img_features])

    # Display recommended images with similarity scores
    st.subheader('Recommended Images with Similarity Scores')
    cols = st.columns(5)
    recommended_images = []
    for i, col in enumerate(cols):
        if i + 1 < len(indices[0]):  # Ensure we have enough recommendations
            img_path = filenames[indices[0][i + 1]]
            recommended_images.append(img_path)
            col.image(img_path)
            similarity_score = 1 - distances[0][i + 1]
            col.write(f"Similarity Score: {similarity_score:.2f}")

    # Save to Favorites
    if st.button('Save to Favorites'):
        st.session_state['favorites'].extend(recommended_images)
        st.success("Images added to favorites!")

    # Display Favorites
    if st.button("View Favorites"):
        if st.session_state['favorites']:
            st.subheader("Favorite Items")
            for fav_img in st.session_state['favorites']:
                st.image(fav_img)
        else:
            st.write("No items in favorites yet.")

    # Social Sharing: Download recommended images
    st.subheader("Download Recommended Images")
    for i in range(1, 6):
        st.download_button(
            label=f"Download Recommendation {i}",
            data=open(filenames[indices[0][i]], 'rb').read(),
            file_name=f"recommendation_{i}.jpg",
            mime="image/jpeg"
        )
