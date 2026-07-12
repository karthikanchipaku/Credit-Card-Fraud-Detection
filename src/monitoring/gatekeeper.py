import json
import subprocess
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
load_dotenv() # Load variables from .env
THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.15))

def check_for_drift(threshold=0.15):
    report_path = "artifacts/reports/data_drift_report.json"
    
    if not os.path.exists(report_path):
        print("❌ Report not found.")
        return

    with open(report_path, "r") as f:
        report_data = json.load(f)

    try:
        # Target the specific metric that identifies how many columns drifted
        metrics = report_data.get('metrics', [])
        drift_metric = next((m for m in metrics if 'DriftedColumnsCount' in m.get('metric_name', '')), None)
        
        if drift_metric:
            # The structure of DriftedColumnsCount has 'drift_share' inside its 'value' dict
            drift_val = drift_metric.get('value', {})
            drift_share = drift_val.get('drift_share', 0)
            
            print(f"📊 Current Drift Share: {float(drift_share):.4f}")

            if float(drift_share) > threshold:
                print("⚠️ Drift threshold exceeded! Initiating retraining...")
                run_evaluation_loop()
            else:
                print("✅ Model performance stable.")
        else:
            print("❌ Could not find DriftedColumnsCount metric.")
            
    except Exception as e:
        print(f"❌ Error parsing drift report: {e}")

def run_evaluation_loop():
    print("🔄 Triggering training pipeline...")
    # Trigger training
    subprocess.run([".\\.venv\\Scripts\\python.exe", "src/training/train.py"], check=True)
    
    # MLflow Evaluation Logic
    client = MlflowClient()
    latest_run = client.search_runs(experiment_ids=["1"], order_by=["start_time DESC"])[0]
    candidate_acc = latest_run.data.metrics.get('accuracy', 0)
    
    try:
        prod_version = client.get_latest_versions("FraudDetectionModel", stages=["Production"])[0]
        prod_run = client.get_run(prod_version.run_id)
        prod_acc = prod_run.data.metrics.get('accuracy', 0)
    except:
        prod_acc = 0
        
    if candidate_acc > prod_acc:
        print(f"🚀 Candidate (Acc: {candidate_acc:.4f}) is better. Promoting!")
        model_uri = f"runs:/{latest_run.info.run_id}/fraud_model"
        mlflow.register_model(model_uri, "FraudDetectionModel")
    else:
        print(f"🛑 Candidate (Acc: {candidate_acc:.4f}) not better than Prod. Discarding.")

if __name__ == "__main__":
    check_for_drift()