import json
import pandas as pd
import io
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session

from api.schemas.transaction import TransactionInput
from src.prediction.predict_pipeline import PredictPipeline
from src.logger.logger import logging
from src.database.database import get_db
from src.database.models import PredictionRecord

router = APIRouter()
predict_pipeline = PredictPipeline()

@router.post("/predict")
async def predict_fraud(transaction: TransactionInput, db: Session = Depends(get_db)):
    try:
        logging.info("Received prediction request via API.")
        
        # 1. Convert the incoming JSON to a DataFrame
        input_data = transaction.model_dump()
        features_df = pd.DataFrame([input_data])
        
        # 2. Make prediction
        prediction = predict_pipeline.predict(features_df)
        
        # 3. Format response
        is_fraud = bool(prediction[0] == 1)
        status = "Fraudulent" if is_fraud else "Legitimate"
        
        logging.info(f"API Prediction successful. Result: {status}")
        
        # 4. --- UPDATED: Save to Database ---
        logging.info("Saving prediction to SQLite database...")
        db_record = PredictionRecord(
            amount=transaction.Amount,
            prediction_result=status,
            is_fraud=is_fraud,
            features=json.dumps(input_data) # <-- NEW: Dump the dict to a string
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        logging.info(f"Saved successfully with Record ID: {db_record.id}")
        
        return {
            "status": "success",
            "prediction": status,
            "is_fraud": is_fraud,
            "record_id": db_record.id
        }
        
    except Exception as e:
        logging.error(f"API Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_prediction_history(db: Session = Depends(get_db), limit: int = 10):
    """
    Fetches the most recent predictions from the database.
    """
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
        return {"history": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV file of transactions, processes them in bulk, and returns the predictions.
    """
    try:
        logging.info(f"Received batch prediction request with file: {file.filename}")
        
        # 1. Read the uploaded file into a Pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Ensure we don't try to predict the target variable if it was accidentally included
        if 'Class' in df.columns:
            features_df = df.drop(columns=['Class'])
        else:
            features_df = df
            
        logging.info(f"Processing {len(features_df)} rows for batch prediction.")
        
        # 2. Make batch predictions
        predictions = predict_pipeline.predict(features_df)
        
        # 3. Append the results to the DataFrame
        df['Prediction'] = ["Fraudulent" if p == 1 else "Legitimate" for p in predictions]
        
        # Convert the resulting DataFrame to a dictionary for the JSON response
        result_dict = df.to_dict(orient='records')
        
        logging.info("Batch prediction successful.")
        
        return {
            "status": "success",
            "total_processed": len(df),
            "results": result_dict
        }
        
    except Exception as e:
        logging.error(f"Batch Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))