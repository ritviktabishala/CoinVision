from pathlib import Path
import joblib
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

SAVE_DIR = Path(__file__).parent.parent / "trained_models"


def load_label_encoder():
    return joblib.load(SAVE_DIR / "label_encoder.joblib")


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    return model


def load_model(label_encoder):
    model = build_model(len(label_encoder.classes_))

    model.load_state_dict(
        torch.load(
            SAVE_DIR / "coin_classifier.pth",
            map_location="cpu"
        )
    )

    model.eval()
    return model


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    transform = get_inference_transform()
    return transform(image).unsqueeze(0)


def predict_image(pil_image, model, label_encoder, device):
    image = preprocess_image(pil_image)
    image = image.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_label = label_encoder.inverse_transform([predicted_idx.item()])[0]
    top_confidence, top_idx = torch.topk(probabilities, 3, dim=1)
    confidences = top_confidence[0]
    indices = top_idx[0]

    top_predictions = [
        {
            "label": label_encoder.inverse_transform([idx.item()])[0],
            "confidence": round(conf.item() * 100, 2)
        }
        for idx, conf in zip(indices, confidences)
    ]

    return {
        "prediction": predicted_label,
        "confidence": round(confidence.item() * 100, 2),
        "top_predictions": top_predictions
    }