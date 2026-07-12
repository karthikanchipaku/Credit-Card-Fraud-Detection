import mlflow
import os
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# This tells MLflow to store data in the project folder, 
# ensuring it's tracked by Git/DVC and accessible by the UI
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("CreditCardFraudDetection")

def train_model(X_train, y_train, X_test, y_test):
    # Set the experiment
    mlflow.set_experiment("CreditCardFraudDetection")
    
    with mlflow.start_run():
        # Initialize and train
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Log Parameters
        mlflow.log_param("model_type", "RandomForest")
        
        # Evaluate
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        
        # Log the Model
        mlflow.sklearn.log_model(model, "fraud_model")
        print(f"✅ Training run logged to MLflow with accuracy: {acc:.4f}")

if __name__ == "__main__":
    # 1. LOAD DATA
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # 2. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. CALL FUNCTION
    train_model(X_train, y_train, X_test, y_test)