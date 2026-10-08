"""Day 3c - Grad-CAM: visualize which region of a document image drove the
tampered/genuine prediction.

Hooks the last convolutional block (layer4) to capture its activations and
the gradient flowing back into them for the predicted class, then combines
them into a heatmap over the original image - this is what will eventually
power the "flagged region" on your verdict card (Day 5/6).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
CHECKPOINT_PATH = MODEL_DIR / "document_forgery_resnet18.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CLASS_NAMES = ["genuine", "tampered"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GradCAM:
    """Minimal Grad-CAM hooked onto a single target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive contributions to the predicted class
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_model() -> torch.nn.Module:
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def overlay_heatmap(original_img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    resized_original = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    return cv2.addWeighted(resized_original, 0.6, heatmap, 0.4, 0)


def run(image_path: str, output_path: str = "gradcam_output.jpg"):
    model = load_model()
    gradcam = GradCAM(model, target_layer=model.layer4[-1])

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, pred_class].item()

    cam = gradcam.generate(input_tensor, class_idx=pred_class)

    original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(original_bgr, cam)
    cv2.imwrite(output_path, overlay)

    print(f"Prediction: {CLASS_NAMES[pred_class]} (confidence: {confidence:.3f})")
    print(f"Saved Grad-CAM overlay to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python 03c_gradcam.py <path_to_image> [output_path]")
        sys.exit(1)
    image_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "gradcam_output.jpg"
    run(image_arg, output_arg)

