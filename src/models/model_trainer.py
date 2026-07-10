import os
import sys
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "models", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train and test array into independent and dependent features")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # 1. Apply SMOTE 
            logging.info(f"Shape of X_train before SMOTE: {X_train.shape}")
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logging.info(f"Shape of X_train after SMOTE: {X_train_resampled.shape}")

            # 2. Train XGBoost
            logging.info("Training XGBoost Classifier...")
            
           # We already balanced the data with SMOTE, so we use standard XGBoost settings
            model = XGBClassifier(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_resampled, y_train_resampled)

            # 3. Evaluate the Model
            logging.info("Predicting on test data and calculating metrics")
            y_pred = model.predict(X_test)
            
            model_f1_score = f1_score(y_test, y_pred)
            
            logging.info(f"XGBoost Model F1 Score: {model_f1_score}")
            print("\nClassification Report (XGBoost):")
            print(classification_report(y_test, y_pred))

            # 4. Save the Model
            logging.info("Saving trained XGBoost model to artifacts directory")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            return model_f1_score, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)