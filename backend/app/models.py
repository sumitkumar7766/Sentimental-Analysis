import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from .database import Base

class SentimentRecord(Base):
    """
    Schema for saving evaluation and prediction runs to support evaluation tracing.
    """
    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, index=True)
    modality = Column(String(50), nullable=False) # "text", "image", "audio", or "multimodal"
    
    # Store textual input or file path references
    text_content = Column(Text, nullable=True)
    file_path = Column(String(255), nullable=True)
    
    # Prediction Results
    predicted_label = Column(String(50), nullable=False) # "Positive", "Negative", "Neutral"
    confidence_score = Column(Float, nullable=False)
    
    # Probability breakdown
    score_positive = Column(Float, nullable=False)
    score_negative = Column(Float, nullable=False)
    score_neutral = Column(Float, nullable=False)
    
    # Modality Contributions (stored as percentages)
    weight_text = Column(Float, nullable=True)
    weight_image = Column(Float, nullable=True)
    weight_audio = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
