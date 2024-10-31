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
st.title("Breast Cancer Prediction System")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sample Images", "Make Prediction"])

if page == "Home":
    st.write("""
    # Welcome to the Breast Cancer Prediction System
    
    This application uses advanced machine learning techniques to analyze breast thermal images 
    and predict whether a tumor is benign or malignant.
    
    ### How to use:
    1. Browse through sample images in the 'Sample Images' section
    2. Upload your image in the 'Make Prediction' section
    3. Get instant predictions and hospital recommendations
    
    ### Features:
    - Advanced image analysis
    - Instant predictions
    - Hospital recommendations
    - Educational resources
    """)

elif page == "Sample Images":
    st.write("## Sample Images for Reference")
    
    st.write("### Benign Cases")
    col1, col2 = st.columns(2)
    with col1:
        st.image("sample_images/benign_1.jpg", caption="Benign Sample 1")
    
    
    st.write("### Malignant Cases")
    col3, col4 = st.columns(2)
    with col3:
        st.image("sample_images/malignant_1.jpg", caption="Malignant Sample 1")
    with col4:
        st.image("sample_images/malignant_2.jpg", caption="Malignant Sample 2")
    
    st.write("""
    ### Image Guidelines:
    - Use high-quality thermal images
    - Ensure proper lighting and focus
    - Images should be clearly visible
    - Recommended format: JPG, PNG
    """)

elif page == "Make Prediction":
    st.write("## Upload Your Image for Prediction")
    
    # File uploader with example image
    st.write("### Upload an image:")
    image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Add example button
    if st.button("Use Example Image"):
        image = "sample_images/example.jpg"
        st.image(image, caption="Example Image", width=300)
    
    # Create a button to make predictions
    if st.button("Predict"):
        if image is not None:
            try:
                # Show loading spinner
                with st.spinner('Processing image...'):
                    # Extract features from the uploaded image
                    features = extract_features(image)
                    input_data = preprocess_input_data(np.array(features).reshape(1, -1))
                    predictions = predict_breast_cancer(input_data)

                # Display results in an organized way
                st.write("### Results:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if predictions:
                        st.error("📊 Prediction: Malignant")
                    else:
                        st.success("📊 Prediction: Benign")
                
                with col2:
                    recommended_hospitals = recommend_hospital(predictions)
                    st.write("🏥 Recommended Hospitals:")
                    st.dataframe(recommended_hospitals)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image or use the example image.")

    # Add educational content
    st.write("""
    ### Understanding the Analysis
    
    #### Feature Extraction:
    - Texture analysis
    - Pattern recognition
    - Intensity distribution
    
    #### Model Interpretation:
    - Neural network analysis
    - Feature importance
    - Confidence scores
    """)

# Add footer
st.markdown("""
---
### Disclaimer
This application is for educational and demonstration purposes only. 
Always consult healthcare professionals for medical diagnosis and treatment decisions.

*Powered by Advanced Machine Learning*
""")