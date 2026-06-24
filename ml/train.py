import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from ml.dataset import CoinDataset

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

datasets_path = Path(__file__).parent.parent / "datasets"
df = pd.read_csv(datasets_path / "train.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["Class"])
num_classes = df["label"].nunique()
img_dir = datasets_path / "train"

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Train Samples: {len(train_df)}")
print(f"Validation Samples: {len(val_df)}")
print(f"Classes: {num_classes}")

train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CoinDataset(train_df, img_dir, train_transform)
val_dataset = CoinDataset(val_df, img_dir, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=5e-4)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100


save_dir = Path(__file__).parent.parent / "trained_models"
save_dir.mkdir(parents=True, exist_ok=True)

num_epochs = 30
best_val_acc = 0.0
best_epoch = 0

for epoch in range(num_epochs):

    if epoch == 7:
        print("\nUnfreezing Layer4...\n")
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5
        )

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_acc = (correct / total) * 100
    val_acc = evaluate(model, val_loader, device)
    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Loss={epoch_loss:.4f} | "
        f"TrainAcc={train_acc:.2f}% | "
        f"ValAcc={val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), save_dir / "coin_classifier.pth")
        print(f"--> New Best Model Saved (Epoch {best_epoch}, ValAcc={best_val_acc:.2f}%)")

joblib.dump(le, save_dir / "label_encoder.joblib")
print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")