"""
modules/violence/violence_detector.py
--------------------------------------
CNN-LSTM violence classifier (Part 11), faithful port from the team branch.

Architecture (MUST match the team's checkpoint exactly):
  ResNet50 (ImageNet) feature extractor (final fc dropped)
    → BiLSTM(input=2048, hidden=256, layers=2, dropout=0.5, bidirectional)
    → head: LayerNorm(512) → Dropout(0.5) → Linear(512,128) → GELU
            → Dropout(0.25) → Linear(128,1)
  Output: single logit per clip; caller applies sigmoid.

The weights file (VIOLENCE_MODEL_PATH) is NOT bundled — obtain it from the team.
If missing/unloadable, `is_ready` is False and the violence worker exits cleanly.
"""

import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class _CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet   = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.5, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(0.5), nn.Linear(512, 128),
            nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 1),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.cnn(x).flatten(1)
        x = x.view(b, t, -1)
        x, _ = self.lstm(x)
        return self.head(x[:, -1]).squeeze(-1)


class ViolenceDetector:
    """Loads the CNN-LSTM checkpoint and scores frame sequences."""

    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self._load(str(model_path))

    def _load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            print(f"[VIOLENCE] Model not found at '{model_path}'. "
                  "Violence detection DISABLED (system runs normally).")
            return
        try:
            model = _CNN_LSTM().to(self.device)
            ckpt  = torch.load(model_path, map_location=self.device)
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            model.eval()
            self.model = model
            print(f"[VIOLENCE] Model loaded from '{model_path}' on {self.device}.")
        except Exception as exc:
            import traceback
            print(f"[VIOLENCE] Failed to load model: {exc}\n{traceback.format_exc()}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """Resize to 224×224 and scale to [0,1] float32 (matches training)."""
        return (cv2.resize(frame, (224, 224)) / 255.0).astype(np.float32)

    def score(self, seq_frames: List[np.ndarray]) -> Optional[float]:
        """Run the model on a sequence of preprocessed frames → violence prob in [0,1]."""
        if not self.is_ready or not seq_frames:
            return None
        try:
            seq    = np.asarray(seq_frames, dtype=np.float32)            # (T, H, W, 3)
            tensor = (torch.tensor(seq).permute(0, 3, 1, 2)             # (T, 3, H, W)
                      .unsqueeze(0).float().to(self.device))            # (1, T, 3, H, W)
            with torch.no_grad():
                return float(torch.sigmoid(self.model(tensor)).item())
        except Exception as exc:
            print(f"[VIOLENCE] Inference error: {exc}")
            return None
