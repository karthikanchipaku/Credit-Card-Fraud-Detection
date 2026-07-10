from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from datetime import datetime
from src.database.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # We will log the transaction amount for easy tracking
    amount = Column(Float)
    
    # The outcome from our ML model
    prediction_result = Column(String)
    is_fraud = Column(Boolean)