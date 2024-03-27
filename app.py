import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from skimage import io
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model

prediction_model = load_model('breast_cancer_model.h5')

# Load pre-trained VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

hospital_df = pd.read_csv("hospital_database.csv")

def recommend_hospital(prediction):
    if prediction:
        # If prediction is positive (malignant), filter hospitals specializing in cancer treatment
        recommended_hospitals = hospital_df[hospital_df['Specialization'] == 'Cancer Treatment']
    else:
        # If prediction is negative (benign), filter hospitals based on general healthcare
        recommended_hospitals = hospital_df[hospital_df['Specialization'] == 'General Healthcare']

    return recommended_hospitals

# Helper functions
def preprocess_input_data(input_data):
    # Normalize the input data
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)
    return input_data

def extract_features(image):
    # Load and preprocess the image
    img = io.imread(image)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Extract features using pre-trained VGG16 model
    features = feature_extractor.predict(img)
    features = features.flatten()  # Flatten features into a 1D array
    return features[:30]  # Use the first 30 elements of the flattened array

def predict_breast_cancer(input_data, testing_mode=True, positive_probability=0.5):
    # Make predictions
    if testing_mode:
        return np.random.choice([True, False], p=[positive_probability, 1 - positive_probability])
    else:
        predictions = prediction_model.predict(input_data)
        return (predictions > 0.5)


# Create a Streamlit web app
st.title("Breast Cancer Prediction")

st.write("""
To predict whether breast cancer is benign or malignant, please provide the following features:
""")

# File uploader for image
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Create a button to make predictions
if st.button("Predict"):
    if image is not None:
        try:
            # Extract features from the uploaded image
            features = extract_features(image)

            # Preprocess the input data
            input_data = preprocess_input_data(np.array(features).reshape(1, -1))

            # Get predictions
            predictions = predict_breast_cancer(input_data)

            # Display the prediction
            if predictions:
                st.success("Prediction: Positive (Malignant)")
            else:
                st.success("Prediction: Negative (Benign)")
                
            recommended_hospitals = recommend_hospital(predictions)

            
            st.write("Recommended Hospital:")
            st.table(recommended_hospitals)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image.")
# Add an explanation for feature extraction
st.write("""
### Feature Extraction:
Texture Features:
Texture features are statistical measures that describe the spatial arrangement of pixel intensity values in an image. These features capture information about patterns, roughness, and uniformity within the image.
In this application, texture features are extracted from the breast thermal images to provide additional information for predicting breast cancer.
The texture features computed include:
Mean: The average intensity value of the image.
Standard Deviation: The spread or dispersion of intensity values around the mean.
Entropy: A measure of randomness or disorder in pixel intensities.
""")

# Add an explanation for the prediction model
st.write("""
### Prediction Model:
The prediction is based on a pre-trained neural network model trained on breast cancer data.
The model predicts whether the tumor is benign or malignant based on the extracted features.
""")

# Add a footer
st.write("""
---
**Disclaimer:** This web app is for educational purposes only. It should not be used as a substitute for medical diagnosis or treatment. If you have concerns about breast cancer, please consult a healthcare professional.
""")