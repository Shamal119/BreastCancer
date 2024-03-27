import numpy as np
from skimage import io, color, feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# List of feature names
feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "SE Radius",
    "SE Texture",
    "SE Perimeter",
    "SE Area",
    "SE Smoothness",
    "SE Compactness",
    "SE Concavity",
    "SE Concave Points",
    "SE Symmetry",
    "SE Fractal Dimension",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension"
]

# Function to extract features from an image
def extract_features(image_path):
    # Load the image
    image = io.imread(image_path)

    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)

    # Example feature: Histogram of Oriented Gradients (HOG)
    hog_features, _ = feature.hog(gray_image, visualize=True)

    # Return a 1D array of features
    return hog_features

# Function to process an image, extract features, train a model, and save the model
def process_image_and_save_model(image_path, model_filename='breast_cancer_model_extraction.h5'):
    # Extract features from the image
    features = extract_features(image_path)

    # Create a dictionary to store features with their names
    extracted_features_dict = dict(zip(feature_names, features))

    # Print and save the extracted features
    for feature_name, value in extracted_features_dict.items():
        print(f"{feature_name}: {value}")

    # Assuming you have a labeled dataset with features and labels
    # Replace this with your actual dataset loading code
    # For simplicity, I'm using a synthetic dataset here
    X = np.random.rand(100, len(features))  # 100 samples, features extracted from the image
    y = np.random.randint(0, 2, 100)  # Binary classification labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (mean=0 and variance=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the trained model
    joblib.dump(model, model_filename)
    print(f"Trained model saved as {model_filename}")

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)




