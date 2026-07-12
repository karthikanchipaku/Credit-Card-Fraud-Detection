import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger.logger import logging
from src.exception.exception import CustomException

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion paths.
    Using dataclass saves us from writing an __init__ method for simple attributes.
    """
    raw_data_path: str = os.path.join("data", "raw", "creditcard.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw data, performs a stratified train-test split, 
        and saves the processed files.
        """
        logging.info("Entered the data ingestion method.")
        try:
            # 1. Read the dataset
            logging.info(f"Reading raw data from {self.ingestion_config.raw_data_path}")
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Dataset read successfully. Shape: {df.shape}")

            # 2. Create the processed directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3. Perform Stratified Train-Test Split
            logging.info("Initiating stratified train-test split.")
            # Stratifying on 'Class' ensures our 80/20 split maintains the exact fraud ratio in both sets
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df['Class']
            )

            # 4. Save the splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion is complete. Train and test sets saved to data/processed/")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error("Exception occurred during data ingestion.")
            raise CustomException(e, sys)