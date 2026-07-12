import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from src.exception.exception import CustomException
from src.logger.logger import logging

class ModelTrainer:
    def __init__(self):
        self.model_name = "XGBoost"

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Starting MLflow tracking for model training")
            
            # 1. Split the incoming arrays into X (features) and y (target)
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # 2. Initialize the model
            model = XGBClassifier()
            
            # 3. Set the MLflow experiment name
            mlflow.set_experiment("Credit_Card_Fraud_Detection")
            
            # 4. Start a tracking run
            with mlflow.start_run():
                
                # Train the model
                logging.info("Training the model...")
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Log Parameters and Metrics to MLflow
                mlflow.log_param("Model_Name", self.model_name)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1_Score", f1)
                
                # Log the actual model file to MLflow
                mlflow.sklearn.log_model(
    model, 
    "model",
    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
)
                
                logging.info(f"Model trained successfully. Accuracy: {accuracy}")
                return accuracy, model
                
        except Exception as e:
            raise CustomException(e, sys)