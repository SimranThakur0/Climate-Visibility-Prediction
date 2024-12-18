import os
import sys
import logging
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

# Set up logging
logger = logging.getLogger('ML_Project_Logger')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# Define constants for directory paths
ARTIFACTS_DIR = "C:\\Users\\shiva\\Climate-Visibility-Prediction\\artifacts"  # Update if necessary
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')

# Function to load the model and preprocessor
def load_model_and_preprocessor():
    try:
        # Load the model and preprocessor
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info(f"Model and preprocessor loaded successfully")
        return model, preprocessor
    except Exception as e:
        raise CustomException(f"Error loading model or preprocessor: {str(e)}", sys)

# Function to preprocess input data
def preprocess_input_data(data, preprocessor):
    try:
        # Drop the 'date' column if it exists
        if 'DATE' in data.columns:
            data = data.drop(columns=['DATE'])
            logger.info("Dropped 'date' column")

        # Ensure the columns in input data match the order of features used in training
        expected_columns = ['DRYBULBTEMPF', 'RelativeHumidity', 'WindSpeed', 'StationPressure', 'WindDirection', 'Precip']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            raise CustomException(f"Missing columns in input data: {missing_columns}", sys)

        # Reorder the columns to match the order of the trained model's features
        data = data[expected_columns]
        
        # Scale the features using the preprocessor (StandardScaler)
        X_scaled = preprocessor.transform(data)
        logger.info("Input data preprocessing completed (features scaled)")

        return X_scaled
    except Exception as e:
        raise CustomException(f"Error during input data preprocessing: {str(e)}", sys)

# Function to make predictions
def make_predictions(model, preprocessed_data):
    try:
        # Predict using the trained model
        predictions = model.predict(preprocessed_data)
        return predictions
    except Exception as e:
        raise CustomException(f"Error making predictions: {str(e)}", sys)

# Main pipeline function
def run_prediction_pipeline(input_data):
    try:
        # Load the model and preprocessor
        model, preprocessor = load_model_and_preprocessor()

        # Preprocess the input data
        preprocessed_data = preprocess_input_data(input_data, preprocessor)

        # Make predictions
        predictions = make_predictions(model, preprocessed_data)

        logger.info(f"Predictions: {predictions}")
        return predictions
    except CustomException as ce:
        logger.error(f"Custom Exception: {ce}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Example usage of the pipeline with input data
if __name__ == "__main__":
    try:
        # Example input data (replace with actual input data for prediction)
        input_data = pd.DataFrame({
            'DRYBULBTEMPF': [25.0],
            'RelativeHumidity': [60.0],
            'WindSpeed': [10.0],
            'StationPressure': [1015.0],
            'WindDirection': [0],
            'Precip': [0]
        })

        # Run the prediction pipeline
        predictions = run_prediction_pipeline(input_data)
        print(f"Predicted Visibility: {predictions}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
