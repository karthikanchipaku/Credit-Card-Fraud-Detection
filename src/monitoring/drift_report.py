import sqlite3
import pandas as pd
import json
import os
from evidently.core.report import Report
from evidently.presets import DataDriftPreset

def generate_drift_report():
    # 1. Load Data
    ref_path = "data/processed/train.csv"
    reference_data = pd.read_csv(ref_path)
    if 'Class' in reference_data.columns:
        reference_data = reference_data.drop(columns=['Class'])

    db_path = os.path.join(os.getcwd(), "prediction_history.db")
    conn = sqlite3.connect(db_path)
    db_data = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    current_features = pd.json_normalize(db_data['features'].apply(json.loads))
    common_cols = reference_data.columns.intersection(current_features.columns)
    current_data = current_features[common_cols]
    reference_data = reference_data[common_cols]
    
    # 2. Run the Report
    print("📊 Running Analysis...")
    report = Report(metrics=[DataDriftPreset()])
    
    # IMPORTANT: Capture the return value of .run()
    # In 0.7+, this returns a Snapshot object
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    # 3. Save the snapshot
    os.makedirs("artifacts/reports", exist_ok=True)
    
    # Use the snapshot to save
    snapshot.save_html("artifacts/reports/data_drift_report.html")
    snapshot.save_json("artifacts/reports/data_drift_report.json")
    
    print("🎉 SUCCESS! Reports saved.")

if __name__ == "__main__":
    generate_drift_report()