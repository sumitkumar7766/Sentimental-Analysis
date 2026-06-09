from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="The input text message to analyze.")

class EmotionRequest(BaseModel):
    text: str = Field(..., description="The input text message to analyze for emotion.")

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]

# Sub-structures
class ScoreBreakdown(BaseModel):
    positive: float
    negative: float
    neutral: float

class AttentionWeight(BaseModel):
    token: str
    weight: float

# Main Responses
class TextAnalysisResponse(BaseModel):
    label: str
    confidence: float
    scores: ScoreBreakdown
    attention_weights: List[AttentionWeight]

class ImageAnalysisResponse(BaseModel):
    label: str
    confidence: float
    scores: ScoreBreakdown
    detected_expression: str

class AudioAnalysisResponse(BaseModel):
    label: str
    confidence: float
    scores: ScoreBreakdown
    waveform: List[float]
    pitch_trend: List[float]

class MultimodalAnalysisResponse(BaseModel):
    label: str
    confidence: float
    scores: ScoreBreakdown
    contributions: Dict[str, float] # e.g. {"text": 45.0, "image": 30.0, "audio": 25.0}
    fusion_type_used: str

class UnifiedPredictionResponse(BaseModel):
    label: str
    confidence: float
    scores: ScoreBreakdown
    contributions: Dict[str, float]
    fusion_type_used: str
    text_expl: Optional[List[AttentionWeight]] = None
    image_expl: Optional[str] = None
    audio_expl: Optional[Dict[str, Any]] = None

# Metrics Dashboard Schemas
class ConfusionMatrixCell(BaseModel):
    actual: str
    predicted: str
    count: int

class ROCPoint(BaseModel):
    fpr: float
    tpr: float
    threshold: float

class TrainingEpochMetric(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    confusion_matrix: List[ConfusionMatrixCell]
    roc_curve: Dict[str, List[ROCPoint]] # "text", "image", "audio", "multimodal"
    training_history: Dict[str, List[TrainingEpochMetric]]

# Dataset Explorer Schemas
class ClassDistribution(BaseModel):
    label: str
    count: int

class DatasetItem(BaseModel):
    id: int
    text_content: Optional[str] = None
    media_url: Optional[str] = None
    true_label: str
    predicted_label: Optional[str] = None

class DatasetMetadata(BaseModel):
    name: str
    category: str # "text" or "multimodal"
    size: int
    description: str
    class_distribution: List[ClassDistribution]
    sample_records: List[DatasetItem]

class RecordResponse(BaseModel):
    id: int
    modality: str
    text_content: Optional[str]
    file_path: Optional[str]
    predicted_label: str
    confidence_score: float
    created_at: datetime

    class Config:
        from_attributes = True
