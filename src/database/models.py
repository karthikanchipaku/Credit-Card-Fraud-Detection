from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from datetime import datetime
from src.database.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    amount = Column(Float)
    prediction_result = Column(String)
    is_fraud = Column(Boolean)
    
    # --- NEW: Store the full transaction payload for Evidently AI ---
    features = Column(Text)