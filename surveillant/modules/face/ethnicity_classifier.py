"""
modules/face/ethnicity_classifier.py
-------------------------------------
Optional ethnicity classification on a face crop (Part 11), merged from the
team branch.

NOTE (ethics): ethnicity inference is demographically sensitive and the model
may not generalize across populations. It is integrated because it was part of
the team's deliverable, but it is gated behind ENABLE_FACE_ANALYSIS and
graceful-disables when its weights are absent — leave it disabled if your
evaluation context calls for that.

Architecture MUST match the team's checkpoint exactly:
  resnet18 backbone, fc replaced with
    Linear(in_features, 128) → ReLU → Dropout(0.5) → Linear(128, num_classes)

The weights file (ETHNICITY_MODEL_PATH) is NOT bundled. If missing, the
classifier disables itself (is_ready == False, predict() → None).
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from typing import Optional, Sequence


class EthnicityClassifier:
    """ResNet18 ethnicity head with graceful-disable loading."""

    def __init__(self, model_path: str, classes: Sequence[str],
                 device: Optional[torch.device] = None):
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = list(classes)
        self.model   = None

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        self._load(str(model_path))

    def _build_model(self) -> nn.Module:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(self.classes)),
        )
        return model

    def _load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            print(f"[ETHNICITY] Model not found at '{model_path}'. "
                  "Ethnicity classification DISABLED (system runs normally).")
            return
        try:
            model = self._build_model()
            ckpt  = torch.load(model_path, map_location=self.device)
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            model.to(self.device).eval()
            self.model = model
            print(f"[ETHNICITY] Model loaded from '{model_path}' | "
                  f"classes={self.classes} | device={self.device}")
        except Exception as exc:
            import traceback
            print(f"[ETHNICITY] Failed to load model: {exc}\n{traceback.format_exc()}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, face_crop: np.ndarray) -> Optional[str]:
        """Run inference on a BGR face crop. Returns a class label or None."""
        if not self.is_ready or face_crop is None or face_crop.size == 0:
            return None
        try:
            rgb    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = torch.argmax(self.model(tensor), dim=1).item()
            return self.classes[pred]
        except Exception as exc:
            print(f"[ETHNICITY] Inference error: {exc}")
            return None
