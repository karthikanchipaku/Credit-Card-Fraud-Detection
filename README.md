# Credit Card Fraud Detection System

A complete end-to-end Machine Learning project designed to detect fraudulent transactions in real-time.

## Tech Stack
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP (Model Explainability)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **DevOps**: Docker, Docker Compose, WSL2

## How to Run
1. Ensure Docker Desktop is running.
2. Clone the repository.
3. Navigate to the project folder.
4. Run the application:
   ```bash
   docker compose -f deployment/docker-compose.yml up --build