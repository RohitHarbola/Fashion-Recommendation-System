# Fashion-Recommendation-System

This project is a Fashion Recommendation System built using Streamlit for the user interface, TensorFlow's ResNet50 model for feature extraction, and k-Nearest Neighbors (k-NN) for finding similar items. Users can upload an image, receive fashion item recommendations, save favorite recommendations, and view/download their favorite items.

Table of Contents

Project Overview

Data and Dataset Explanation

Algorithms and Techniques

How to Run the Project

Explanation of Features

Project Overview

This fashion recommendation system takes an image uploaded by the user, extracts its visual features using a deep learning model, and finds similar images from a preprocessed dataset. It helps users discover similar fashion items based on visual similarity. Additionally, the user can save their favorite recommendations and view them at any time during the session.

Data and Dataset Explanation

The dataset used for this project comprises fashion item images, each representing different categories (clothing, accessories, etc.). The images are stored locally, and their features have been precomputed for faster recommendations.

Image Dataset: Contains thousands of fashion images.

Image Features: Extracted using the ResNet50 model (without the fully connected top layer) and stored in Image_features.pkl. This file contains the high-dimensional vectors representing image features.

Filenames: Paths to the images stored in filenames.pkl.

Why This Dataset?

The dataset provides diverse fashion images that enable the system to generalize recommendations across a wide variety of fashion categories. Precomputing the features allows real-time similarity calculations, enhancing performance.

Algorithms and Techniques

ResNet50 Model (Feature Extraction):

Why Used: ResNet50 is a deep convolutional neural network known for its strong performance in image classification and feature extraction. We use it for extracting high-level visual features from the uploaded image.

Modification: The top layer is excluded, and a GlobalMaxPool2D layer is added to obtain a compact feature vector.

k-Nearest Neighbors (k-NN) (Similarity Search):

Why Used: k-NN is simple, effective, and appropriate for finding similar items when precomputed features are available. It calculates distances between feature vectors to find the most visually similar images.

Metric: Euclidean distance is used to compute similarity.

How to Run the Project

Prerequisites

Python 3.7 or above

Install the following libraries:

pip install numpy pickle tensorflow streamlit scikit-learn

Steps to Run

Clone or download the project.

Place the Image_features.pkl and filenames.pkl files in the project directory.

Store all fashion images in the appropriate directory.

Run the Streamlit application:

streamlit run app.py

The application will open in your default browser.

Explanation of Features

1. Upload an Image

Users can upload a fashion item image using the file uploader.

2. Recommended Images

The system extracts features from the uploaded image and finds the most visually similar items using k-NN.

Similarity Scores: Displayed to show how close each recommended image is to the uploaded image.

3. Save to Favorites

Users can click the "Save to Favorites" button to add recommended images to a favorite list.

4. View Favorites

Clicking "View Favorites" shows all previously saved items during the current session.

5. Download Options

Users can download the recommended images.

Why These Techniques Were Used

ResNet50 was chosen due to its proven ability to capture complex visual features, making it suitable for visual similarity tasks.

k-NN is a straightforward yet effective algorithm for similarity search when working with fixed feature vectors. Its performance is sufficient for smaller datasets and offline recommendations.
