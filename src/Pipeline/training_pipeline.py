
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Example DataFrame for training
df = pd.read_csv('C:\\Users\\shiva\\Climate-Visibility-Prediction\\artifacts\\train.csv')  # Load your dataset

# Feature columns and target column
X = df.drop(columns=['VISIBILITY', 'DATE'])  # Drop target and any irrelevant features like 'DATE'
y = df['VISIBILITY']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the preprocessor (Scaler, Encoder, etc.)
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Train a simple model (Linear Regression for simplicity)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model, preprocessor, and feature names
with open('artifacts/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
logger.info("Model saved successfully.")

with open('artifacts/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
logger.info("Preprocessor (Scaler) saved successfully.")

with open('artifacts/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
logger.info("Feature names saved successfully.")
