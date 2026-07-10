import sys
from src.logger.logger import logging
from src.data.data_ingestion import DataIngestion
from src.preprocessing.data_transformation import DataTransformation
from src.models.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("--- Starting Model Training Pipeline ---")
        
        # 1. Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print("Data Ingestion successful!")
        
        # 2. Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        print("Data Transformation successful!")
        
        # 3. Model Training (Now with XGBoost)
        model_trainer = ModelTrainer()
        f1, model_path = model_trainer.initiate_model_training(train_arr, test_arr)
        
        print(f"\nXGBoost Training successful!")
        print(f"New Model F1 Score: {f1:.4f}")
        print(f"Model saved at: {model_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")