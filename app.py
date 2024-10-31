import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Constants
SAMPLE_IMAGES_DIR = 'sample_images'

# Create sample_images directory if it doesn't exist
if not os.path.exists(SAMPLE_IMAGES_DIR):
    os.makedirs(SAMPLE_IMAGES_DIR)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load or create dummy hospital data
@st.cache_data
def load_hospital_data():
    try:
        return pd.read_csv("hospital_database.csv")
    except:
        return pd.DataFrame({
            'Hospital Name': ['City Hospital', 'Cancer Care Center', 'General Hospital'],
            'Specialization': ['General Healthcare', 'Cancer Treatment', 'General Healthcare'],
            'Location': ['Downtown', 'Uptown', 'Midtown'],
            'Contact': ['123-456-7890', '234-567-8901', '345-678-9012']
        })

hospital_df = load_hospital_data()

# Helper functions
def create_placeholder_image(text="Sample", size=(300, 300), color='gray'):
    """Create a placeholder image with text"""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', size, color=color)
    d = ImageDraw.Draw(img)
    d.text((size[0]/3, size[1]/2), text, fill="white")
    return img

@st.cache_data
def load_sample_image(image_type):
    """Load or create sample image"""
    try:
        path = f"sample_images/{image_type}.jpg"
        if os.path.exists(path):
            return Image.open(path)
        else:
            color = 'green' if 'benign' in image_type else 'red'
            return create_placeholder_image(image_type, color=color)
    except Exception as e:
        return create_placeholder_image(f"Error: {image_type}")

def recommend_hospital(prediction):
    """Recommend hospitals based on prediction"""
    if prediction:
        return hospital_df[hospital_df['Specialization'] == 'Cancer Treatment']
    return hospital_df[hospital_df['Specialization'] == 'General Healthcare']

def make_prediction(image):
    """Make prediction (demo version)"""
    # In production, replace this with actual model prediction
    return np.random.choice([True, False], p=[0.3, 0.7])

# Main app
st.title("Breast Cancer Prediction System")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sample Images", "Make Prediction"])

# Home page
if page == "Home":
    st.write("""
    # Welcome to the Breast Cancer Prediction System
    
    This application demonstrates the potential of AI in medical diagnosis, 
    specifically for breast cancer prediction using image analysis.
    
    ### Features:
    - 🔍 Image Analysis
    - 🏥 Hospital Recommendations
    - 📊 Instant Results
    
    ### How to use:
    1. Browse sample images in the 'Sample Images' section
    2. Upload your image in the 'Make Prediction' section
    3. Get instant predictions and hospital recommendations
    """)

# Sample Images page
elif page == "Sample Images":
    st.write("## Sample Images Reference")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Benign Cases")
        st.image(load_sample_image("benign_1"), caption="Benign Sample 1")
        
    
    with col2:
        st.subheader("Malignant Cases")
        st.image(load_sample_image("malignant_1"), caption="Malignant Sample 1")
       

# Prediction page
else:
    st.write("## Upload Your Image for Prediction")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
      
    
    with col2:
        if st.button("Make Prediction") and (uploaded_file is not None or 'image' in locals()):
            with st.spinner('Analyzing image...'):
                prediction = make_prediction(image)
                
                if prediction:
                    st.error("🔴 Prediction: Malignant")
                else:
                    st.success("🟢 Prediction: Benign")
                
                st.write("### Recommended Hospitals:")
                st.dataframe(recommend_hospital(prediction))

# Footer
st.markdown("""
---
### Disclaimer
This is a demonstration application. Always consult healthcare professionals for medical diagnosis.

""")