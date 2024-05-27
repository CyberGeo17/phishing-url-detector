# phishingurldetector
A Python URL Detector program I made that detects phishing URLs using machine learning algorithms 

This project is a Python program that uses a Random Forest classifier to detect phishing URLs. It extracts features from URLs and classifies them as either phishing or legitimate.

# How to Use

# Training the Model
If the model file (`phishing_model.pkl`) does not exist, the program will train a new model using a provided dataset.

# Predicting URLs
The program accepts URLs as command-line arguments and predicts whether they are phishing or legitimate.

# Requirements
- Python 3
- pandas
- scikit-learn
- joblib

## Example
python phishing_detectt.py http://example.com http://phishing.com
