import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from ..database import get_db
from ..models import SentimentRecord
from ..schemas import (
    UnifiedPredictionResponse,
    AttentionWeight,
    ScoreBreakdown,
    EmotionRequest,
    EmotionResponse
)
from ..services.preprocessing import preprocess_text, preprocess_image, preprocess_audio
from ..services.inference import InferenceService
from ..services.explainability import ExplainabilityService

router = APIRouter(prefix="/predict", tags=["prediction"])
inference_service = InferenceService()

# Directory for uploaded temp assets
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("", response_model=UnifiedPredictionResponse)
def predict_unified(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    fusion_type: str = Form("hybrid"),
    db: Session = Depends(get_db)
):
    """
    Unified multimodal prediction API.
    Accepts text, image, and/or audio inputs, runs feature extraction using the weights models,
    and returns fused sentiment scores, modality contributions, and explanation components.
    """
    img_path = None
    if image is not None:
        img_path = os.path.join(UPLOAD_DIR, f"predict_{image.filename}")
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
    aud_path = None
    if audio is not None:
        aud_path = os.path.join(UPLOAD_DIR, f"predict_{audio.filename}")
        with open(aud_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

    # 1. Run Preprocessing (simulated or real extraction trace)
    if text:
        _ = preprocess_text(text)
    if img_path:
        _ = preprocess_image(img_path)
    if aud_path:
        _ = preprocess_audio(aud_path)

    # 2. Run Multimodal Inference
    label, confidence, scores, contributions = inference_service.analyze_multimodal(
        text=text, 
        image_path=img_path, 
        audio_path=aud_path, 
        fusion_type=fusion_type
    )

    # 3. Generate individual modality explanation packages
    text_expl = None
    if text:
        lime_results = ExplainabilityService.run_lime_text(text, label)
        text_expl = [
            AttentionWeight(token=item["word"], weight=item["weight"]) 
            for item in lime_results
        ]

    image_expl = None
    if img_path:
        # Get detected expression
        _, _, _, image_expl = inference_service.analyze_image(img_path)

    audio_expl = None
    if aud_path:
        # Get waveform and pitch trend
        _, _, _, waveform, pitch_trend = inference_service.analyze_audio(aud_path)
        audio_expl = {
            "waveform": waveform,
            "pitch_trend": pitch_trend
        }

    # 4. Save to Database
    record = SentimentRecord(
        modality="unified",
        text_content=text,
        file_path=f"{img_path if img_path else ''};{aud_path if aud_path else ''}",
        predicted_label=label,
        confidence_score=confidence,
        score_positive=scores["positive"],
        score_negative=scores["negative"],
        score_neutral=scores["neutral"],
        weight_text=contributions.get("text", 33.3),
        weight_image=contributions.get("image", 33.3),
        weight_audio=contributions.get("audio", 33.4)
    )
    db.add(record)
    db.commit()

    return UnifiedPredictionResponse(
        label=label,
        confidence=confidence,
        scores=ScoreBreakdown(**scores),
        contributions=contributions,
        fusion_type_used=fusion_type,
        text_expl=text_expl,
        image_expl=image_expl,
        audio_expl=audio_expl
    )

@router.post("/emotion", response_model=EmotionResponse)
def predict_emotion(
    request: EmotionRequest,
    db: Session = Depends(get_db)
):
    """
    Predicts emotional category from raw text using text_model_emotion weights.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
        
    emotion, confidence, probabilities = inference_service.analyze_emotion(request.text)
    
    # Save a record to DB (since it's a request, audit logging is recommended)
    record = SentimentRecord(
        modality="text_emotion",
        text_content=request.text,
        predicted_label=emotion,
        confidence_score=confidence,
        score_positive=probabilities.get("Happiness", 0.0),
        score_negative=probabilities.get("Sadness", 0.0) + probabilities.get("Anger", 0.0) + probabilities.get("Fear", 0.0) + probabilities.get("Disgust", 0.0),
        score_neutral=probabilities.get("Neutral", 0.0) + probabilities.get("Surprise", 0.0),
        weight_text=100.0,
        weight_image=0.0,
        weight_audio=0.0
    )
    db.add(record)
    db.commit()
    
    return EmotionResponse(
        emotion=emotion,
        confidence=confidence,
        probabilities=probabilities
    )
