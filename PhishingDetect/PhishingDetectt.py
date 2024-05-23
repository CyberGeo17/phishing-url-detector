import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import warnings
import joblib
import argparse

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to train the model
def train_model():
    # Load dataset
    data = pd.read_csv('MachineLearning/dataset_phishing.csv')

    # Adjust these column names based on your dataset
    url_column = 'url'  # Column name for URLs
    label_column = 'status'  # Column name for labels

    # Check for missing values in the status column
    print("\nChecking for missing values in the status column:")
    print(data[label_column].isnull().sum())

    # Drop rows with missing values in the status column
    data = data.dropna(subset=[label_column])

    # Map the labels to binary values (0 for legitimate, 1 for phishing)
    data[label_column] = data[label_column].map({'legitimate': 0, 'phishing': 1})

    # Ensure there are no unmapped values
    print("\nUnique values in the status column after mapping:")
    print(data[label_column].unique())

    # Function to extract features from a URL
    def extract_features(url):
        parsed_url = urlparse(url)
        features = []
        features.append(len(url))  # Length of URL
        features.append(url.count('-'))  # Number of dashes in URL
        features.append(url.count('@'))  # Number of '@' symbols in URL
        features.append(url.count('.'))  # Number of dots in URL
        features.append(url.count('/'))  # Number of slashes in URL
        features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  # Presence of IP address
        features.append(len(parsed_url.netloc.split('.')) - 2)  # Number of subdomains
        return features

    # Apply feature extraction to the dataset
    X = data[url_column].apply(extract_features)
    X = pd.DataFrame(X.tolist(), columns=['length', 'num_dashes', 'num_at', 'num_dots', 'num_slashes', 'has_ip', 'num_subdomains'])
    y = data[label_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    # Cross-Validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {cv_scores.mean():.2f}")

    # Save the model to a file
    joblib.dump(clf, 'phishing_model.pkl')

    print("Model training complete and saved as phishing_model.pkl")

# Function to extract features from a URL
def extract_features(url):
    parsed_url = urlparse(url)
    features = []
    features.append(len(url))  # Length of URL
    features.append(url.count('-'))  # Number of dashes in URL
    features.append(url.count('@'))  # Number of '@' symbols in URL
    features.append(url.count('.'))  # Number of dots in URL
    features.append(url.count('/'))  # Number of slashes in URL
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  # Presence of IP address
    features.append(len(parsed_url.netloc.split('.')) - 2)  # Number of subdomains
    return features

# Function to predict if a URL is phishing or legitimate
def predict_url(url, model):
    features = extract_features(url)
    prediction = model.predict([features])
    return "Phishing" if prediction[0] else "Legitimate"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Check if URLs are phishing or legitimate.")
parser.add_argument('urls', metavar='URL', type=str, nargs='+', help='URLs to check')
args = parser.parse_args()

# Check if the model exists
if not os.path.exists('phishing_model.pkl'):
    print("Model not found. Training model...")
    train_model()

# Load the trained model from a file
print("Loading the model...")
clf_loaded = joblib.load('phishing_model.pkl')

# Check each URL provided as argument and print the result
for url in args.urls:
    result = predict_url(url, clf_loaded)
    print(f"The URL '{url}' is {result}")
