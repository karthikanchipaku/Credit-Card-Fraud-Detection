from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="REST API for predicting fraudulent credit card transactions.",
    version="1.0.0"
)

# Allow Cross-Origin Requests (CORS) so our Streamlit frontend can talk to this API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our prediction route
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Fraud Detection API is running."}