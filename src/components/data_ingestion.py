import sys
import os

# Setting project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from src.logger import setup_logger  # Ensure this is correctly imported
logger = setup_logger()
from src.exception import CustomException  # Ensure this is correctly imported
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Data ingestion configuration class
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(project_root, 'artifacts', 'raw.csv')
    train_data_path: str = os.path.join(project_root, 'artifacts', 'train.csv')
    test_data_path: str = os.path.join(project_root, 'artifacts', 'test.csv')
    test_size: float = 0.30  # This can be adjusted more easily now

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info('Starting data ingestion process.')

        try:
            # Ensure dataset file path is correct
            data_file_path = os.path.join(project_root, 'Dataset', 'NOAA-JKF-AP.csv')
            if not os.path.exists(data_file_path):
                logger.error(f"Data file not found at path: {data_file_path}")
                raise FileNotFoundError(f"Data file not found at path: {data_file_path}")

            # Read dataset
            df = pd.read_csv(data_file_path)
            logger.info(f'Dataset loaded successfully with shape: {df.shape}')

            # Create directories for artifacts if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info(f'Raw data saved at: {self.ingestion_config.raw_data_path}')

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=self.ingestion_config.test_size, random_state=42)
            logger.info(f'Data split into train (shape={train_set.shape}) and test (shape={test_set.shape}) sets.')

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info(f'Train data saved at: {self.ingestion_config.train_data_path}')
            logger.info(f'Test data saved at: {self.ingestion_config.test_data_path}')

            logger.info('Data Ingestion process completed successfully.')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logger.exception('Exception occurred during the data ingestion process.')
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")
