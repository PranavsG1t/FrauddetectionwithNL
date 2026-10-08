"""Standalone worker: load the document-forgery ResNet18 and classify one
random test-split image. Run as a subprocess so this never shares a
process with joblib/shap/lightgbm - avoids a native-library interaction
that causes torch.load() to hang when run in the same process as those.
Prints exactly one JSON line to stdout; everything else goes to stderr.
"""
import json
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

DEVICE = torch.device("cpu")
IMG_SIZE = 224

doc_eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    print("[worker] building architecture...", file=sys.stderr, flush=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    print("[worker] loading checkpoint...", file=sys.stderr, flush=True)
    model.load_state_dict(torch.load("outputs/models/document_forgery_resnet18.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    print("[worker] reading manifest...", file=sys.stderr, flush=True)
    manifest = pd.read_csv("outputs/document_dataset/manifest.csv")
    test_rows = manifest[manifest["split"] == "test"]
    row = test_rows.sample(1).iloc[0]

    print(f"[worker] running inference on {row['path']}...", file=sys.stderr, flush=True)
    img = Image.open(row["path"]).convert("RGB")
    input_tensor = doc_eval_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    prediction = "tampered" if pred_idx == 1 else "genuine"
    result = {"prediction": prediction, "confidence": confidence,
              "path": row["path"], "actual_label": int(row["label"])}
    print(json.dumps(result))  # the one line main process reads from stdout


if __name__ == "__main__":
    main()
    