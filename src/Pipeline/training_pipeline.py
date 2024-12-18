import os
import sys
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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
TRAINING_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'train.csv')
TESTING_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'test.csv')

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        raise CustomException(f"Error loading data from {file_path}: {str(e)}", sys)

# Function to preprocess the data (dropping date column if exists, scaling)
def preprocess_data(data):
    try:
        # Drop the 'date' column if it exists
        if 'DATE' in data.columns:
            data = data.drop(columns=['DATE'])
            logger.info("Dropped 'date' column")

        # Separate features and target
        X = data.drop(columns=['VISIBILITY'])
        y = data['VISIBILITY']

        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Data preprocessing completed (features scaled)")

        return X_scaled, y
    except Exception as e:
        raise CustomException(f"Error during data preprocessing: {str(e)}", sys)

# Function to define and train the model
def train_model(X_train, y_train):
    try:
        # Define RandomForestRegressor
        model = RandomForestRegressor()

        # Hyperparameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        raise CustomException(f"Error during model training: {str(e)}", sys)

# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate performance metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)

        # Log the results
        logger.info(f"Train R² Score: {train_r2}")
        logger.info(f"Test R² Score: {test_r2}")
        logger.info(f"Train MAE: {train_mae}")
        logger.info(f"Test MAE: {test_mae}")
        logger.info(f"Mean Squared Error: {mse}")

    except Exception as e:
        raise CustomException(f"Error during model evaluation: {str(e)}", sys)

# Main pipeline function
def run_pipeline():
    try:
        # Load training and testing data
        train_data = load_data(TRAINING_DATA_PATH)
        test_data = load_data(TESTING_DATA_PATH)

        # Preprocess the data
        X_train, y_train = preprocess_data(train_data)
        X_test, y_test = preprocess_data(test_data)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_train, y_train, X_test, y_test)

        logger.info("Model training and evaluation completed successfully")
    except CustomException as ce:
        logger.error(f"Custom Exception: {ce}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Run the pipeline
if __name__ == "__main__":
    run_pipeline()
