from sqlalchemy.orm import Session
from app.models.prediction import Prediction

from ml.inference import predict_image

def save_prediction(db: Session, filename: str, prediction: str, confidence: float):
    prediction_obj = Prediction(
        filename = filename,
        prediction = prediction,
        confidence = confidence
    )

    db.add(prediction_obj)
    db.commit()
    db.refresh(prediction_obj)

    return prediction_obj

def get_predictions(db: Session, limit: int = 50):
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )

def predict_coin(image, model, label_encoder, device):
    result = predict_image(image, model, label_encoder, device)

    return result