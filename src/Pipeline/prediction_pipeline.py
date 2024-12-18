import pandas as pd
import pickle
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class PredictionPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            # Load the trained model
            with open('artifacts/trained_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully.")

            # Load the preprocessor (Scaler, Encoder, etc.)
            with open('artifacts/scaler.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded successfully.")

            # Load feature names
            with open('artifacts/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            logger.info(f"Feature names loaded: {self.feature_names}")

        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def preprocess_input_data(self, input_data: pd.DataFrame) -> np.ndarray:
        try:
            # Ensure input data has the required columns
            missing_cols = [col for col in self.feature_names if col not in input_data.columns]
            for col in missing_cols:
                input_data[col] = 0  # Fill missing columns with 0
                logger.info(f"Added missing column: {col}")

            # Reorder input data columns to match the training feature order
            input_data = input_data[self.feature_names]
            logger.info(f"Reordered input data columns: {input_data.columns.tolist()}")

            # Apply the same preprocessing (scaling)
            preprocessed_data = self.preprocessor.transform(input_data)
            logger.info("Preprocessed data.")
            return preprocessed_data

        except Exception as e:
            logger.error(f"Error during input data preprocessing: {e}")
            raise

    def predict(self, input_data: pd.DataFrame):
        try:
            # Preprocess input data
            preprocessed_data = self.preprocess_input_data(input_data)

            # Make predictions using the trained model
            predictions = self.model.predict(preprocessed_data)
            logger.info(f"Predicted Visibility: {predictions}")
            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None


# Example usage of the Prediction Pipeline
if __name__ == "__main__":
    pipeline = PredictionPipeline()

    # Example input data for prediction
    input_data = pd.DataFrame({
        'DRYBULBTEMPF': [75],
        'WETBULBTEMPF': [70],
        'DewPointTempF': [65],
        'RelativeHumidity': [80],
        'WindSpeed': [12],
        'WindDirection': [100],
        'StationPressure': [29.85],
        'SeaLevelPressure': [30.00],
        'Precip': [0.01]
    })

    # Call the predict method
    prediction = pipeline.predict(input_data)
    print(f"Predicted Visibility: {prediction}")
