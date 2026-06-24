import torch
from ml.inference import predict_image, preprocess_image
from ml.gradcam import generate_gradcam

from app.schemas.prediction import AnalysisResponse
from app.services.image_service import image_to_base64

def analyze_coin(image, model, label_encoder, device) -> AnalysisResponse:

    prediction_result = predict_image(image, model, label_encoder, device)
    
    original_image = image.convert("RGB")
    image_tensor = preprocess_image(original_image).to(device)
    
    target_layer = model.layer4 
    
    predicted_label = prediction_result["prediction"]
    class_idx = int(label_encoder.transform([predicted_label])[0])

    overlay = generate_gradcam(
        image_tensor=image_tensor,
        original_image=original_image,  
        model=model,
        target_layer=target_layer,
        class_idx=class_idx
    )
    
    gradcam_b64 = image_to_base64(overlay)
    
    return AnalysisResponse(
        prediction=prediction_result["prediction"],
        confidence=prediction_result["confidence"],
        top_predictions=prediction_result["top_predictions"],
        gradcam_image=f"data:image/jpeg;base64,{gradcam_b64}"
    )