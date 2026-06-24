from io import BytesIO
from fastapi import APIRouter, UploadFile, HTTPException, status, Depends
import torch
from sqlalchemy.orm import Session
from PIL import Image

from ml.inference import load_model, load_label_encoder

from app.schemas.prediction import AnalysisResponse, PredictionResponse, PredictionHistoryItem
from app.database.dependencies import get_db
from app.database.base import Base
from app.database.db import engine
from app.services.prediction_service import get_predictions, save_prediction, predict_coin
from app.services.analysis_service import analyze_coin

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)

Base.metadata.create_all(bind=engine)

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    label_encoder = load_label_encoder()
    model = load_model(label_encoder)
    model = model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to initialize core ML artifacts on startup: {str(e)}")

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile, db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Must be one of: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        analysis_result = analyze_coin(
            image,
            model,
            label_encoder,
            device
        )

        save_prediction(
            db,
            filename=file.filename,
            prediction=analysis_result.prediction,
            confidence=analysis_result.confidence
        )
        
        return analysis_result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference execution: {str(e)}"
        )

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile, db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Must be one of: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        prediction_result = predict_coin(
            image,
            model,
            label_encoder,
            device
        )

        save_prediction(
            db,
            filename=file.filename,
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"]
        )
        
        return prediction_result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference execution: {str(e)}"
        )

@router.get("/", response_model=list[PredictionHistoryItem])
def get_prediction_history_endpoint(db: Session = Depends(get_db)):
    return get_predictions(db)