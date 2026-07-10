import sys
import os
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Define paths to our saved artifacts
        self.model_path = os.path.join("artifacts", "models", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "models", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Loads artifacts, scales the incoming features, and returns a prediction.
        """
        try:
            logging.info("Loading preprocessor and model artifacts...")
            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)
            
            logging.info("Applying preprocessing to input features...")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making prediction...")
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)