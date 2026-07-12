# Credit Card Fraud Detection System (MLOps Edition)

A professional-grade, end-to-end Machine Learning pipeline for real-time fraud detection. This project demonstrates an automated MLOps lifecycle featuring drift monitoring, model governance, and automated retraining.

📊 Model Governance & Experiment Tracking
This project uses MLflow to track every training experiment, ensuring full auditability.

How to view experiments:
After running the training pipeline (python src/training/train.py), launch the tracking server:

mlflow ui
Open your browser to: http://localhost:5000   

## 🚀 Key Features
* **Automated Drift Detection**: Uses Evidently AI to monitor data distribution shifts.
* **Autonomous Gatekeeper**: A custom decision engine that triggers retraining only when drift thresholds are breached.
* **Model Governance**: MLflow-backed model registry for experiment tracking and A/B testing models before promotion.
* **Data Provenance**: DVC-integrated for robust dataset versioning.

## 🏗️ Architecture Overview


*The workflow automates the transition from raw transaction data to a governed production model.*

## 🛠 Tech Stack
- **ML & Explainability**: Scikit-Learn, XGBoost, SHAP
- **Deployment**: FastAPI, Streamlit, Docker
- **MLOps & Governance**: MLflow, Evidently AI, DVC

## 🏃 Getting Started

### Prerequisites
- Docker Desktop
- Python 3.10+
- DVC installed

### Local Development
1. **Initialize Data**: `dvc pull`
2. **Launch Monitoring**: `python -m src.monitoring.drift_report`
3. **Run Automation**: `python src/monitoring/gatekeeper.py`
4. **Deploy**:
   ```bash
   docker compose -f deployment/docker-compose.yml up --build

   