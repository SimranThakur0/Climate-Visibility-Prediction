import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
import logging
from src.exception import CustomException

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_Project_Logger")

# Correcting the path for artifacts
ARTIFACTS_DIR = r"C:\Users\shiva\Climate-Visibility-Prediction\artifacts"

# Function to load data
def load_data(file_path):
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error occurred while loading data: {e}")
        raise CustomException(e, sys)

# Function to preprocess data (drop 'date' column and handle missing values if needed)
def preprocess_data(data):
    try:
        logger.info("Preprocessing data")
        data = data.drop(columns=['DATE'])  # Dropping the 'DATE' column
        # Handling missing values (if any)
        data = data.dropna()  # or use .fillna() based on your needs
        logger.info("Data preprocessing completed")
        return data
    except Exception as e:
        logger.error(f"Error occurred during data preprocessing: {e}")
        raise CustomException(e, sys)

# Function to train the model
def train_model(data):
    try:
        logger.info("Starting training process")
        
        # Split data into features (X) and target (y)
        X = data.drop(columns=['VISIBILITY'])
        y = data['VISIBILITY']
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize RandomForestRegressor
        model = RandomForestRegressor(random_state=42)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        train_predictions = grid_search.best_estimator_.predict(X_train)
        test_predictions = grid_search.best_estimator_.predict(X_test)

        # Calculate R² score
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        # Calculate Mean Absolute Error (MAE)
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)

        # Log the results
        logging.info(f"Train R² Score: {train_r2}")
        logging.info(f"Test R² Score: {test_r2}")
        logging.info(f"Train MAE: {train_mae}")
        logging.info(f"Test MAE: {test_mae}")

        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Mean Squared Error: {mse}")
        return best_model
    
    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise CustomException(e, sys)

# Main execution
if __name__ == "__main__":
    try:
        # Load data
        train_data = load_data(os.path.join(ARTIFACTS_DIR, 'train.csv'))
        test_data = load_data(os.path.join(ARTIFACTS_DIR, 'test.csv'))

        # Preprocess data
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

        # Train the model
        best_model = train_model(train_data)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error occurred in model training process: {e}")
