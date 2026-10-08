"""Day 3b - Fine-tune ResNet18 on the genuine/tampered document dataset.

Two-phase transfer learning:
  Phase 1: freeze the pretrained backbone, train only the new classification
           head (fast, teaches the new 2-class problem without touching the
           generic ImageNet features).
  Phase 2: unfreeze the last residual block (layer4) and fine-tune it + the
           head together at a much lower learning rate (adapts the
           task-specific late features to document-forgery patterns without
           destroying the pretrained weights).
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "document_dataset"
MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 5        # phase 1: frozen backbone
FINE_TUNE_EPOCHS = 5   # phase 2: unfrozen layer4
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-4    # much lower than HEAD_LR - avoid destroying pretrained weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DocumentDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform):
        self.paths = manifest["path"].tolist()
        self.labels = manifest["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze everything first

    # Replace the classification head - always trainable, 2 classes (genuine/tampered)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def unfreeze_layer4(model: nn.Module):
    for param in model.layer4.parameters():
        param.requires_grad = True


def run_epoch(model, loader, criterion, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_loader = DataLoader(
        DocumentDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        DocumentDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        DocumentDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False
    )

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 0.75]).to(DEVICE))  # genuine is the minority class(1:2), so upweight it

    # Phase 1 - train only the head, backbone frozen
    print("\nPhase 1: training classification head (backbone frozen)...")
    head_optimizer = torch.optim.Adam(model.fc.parameters(), lr=HEAD_LR)
    for epoch in range(HEAD_EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, head_optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        print(f"  Epoch {epoch + 1}/{HEAD_EPOCHS} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    # Phase 2 - unfreeze layer4, fine-tune at a much lower LR
    print("\nPhase 2: fine-tuning layer4 + head (low LR)...")
    unfreeze_layer4(model)
    fine_tune_params = list(model.layer4.parameters()) + list(model.fc.parameters())
    fine_tune_optimizer = torch.optim.Adam(fine_tune_params, lr=FINE_TUNE_LR)
    for epoch in range(FINE_TUNE_EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, fine_tune_optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        print(f"  Epoch {epoch + 1}/{FINE_TUNE_EPOCHS} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    print("\nEvaluating on held-out test set...")
    test_metrics = evaluate(model, test_loader)
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")

    checkpoint_path = MODEL_DIR / "document_forgery_resnet18.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

