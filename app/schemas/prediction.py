from pydantic import BaseModel, ConfigDict
from datetime import datetime

class TopPrediction(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    
    top_predictions: list[TopPrediction]

class AnalysisResponse(BaseModel):
    prediction: str
    confidence: float

    top_predictions: list[TopPrediction]

    gradcam_image: str

class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
