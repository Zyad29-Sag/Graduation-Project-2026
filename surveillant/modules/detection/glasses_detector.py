"""
modules/detection/glasses_detector.py
--------------------------------------
Glasses classification on a face crop, using a PyTorch model (Part 11).

Merged from the team branch. The model path comes from config
(GLASSES_MODEL_PATH); the file is NOT bundled — obtain it from the team. If the
file is missing or fails to load, the detector disables itself gracefully
(`is_ready == False`, `predict()` returns None) and the rest of the system runs
normally.

The checkpoint may be:
  • a full model object  (torch.save(model, path)), or
  • a state-dict         (torch.save(sd, path) or {"model_state": sd}).

Output-layer size auto-detected at load time:
  1 output  → binary sigmoid (glasses / no glasses)
  2 outputs → 2-class softmax, class 1 = glasses
  N outputs → softmax over inferred / indexed class names

Input : face crop (BGR numpy array), resized to 224×224 internally.
Output: a label string ("Glasses", "No Glasses", ...) or None if uncertain.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from typing import Optional

try:
    from config.settings import GLASSES_CONF_THRESHOLD as _CFG_CONF
except Exception:
    _CFG_CONF = 0.50


# Default class names for common output sizes.
_DEFAULT_CLASS_NAMES = {
    1: ["No Glasses", "Glasses"],          # binary sigmoid
    2: ["No Glasses", "Glasses"],          # 2-class softmax
    3: ["No Glasses", "Glasses", "Sunglasses"],
    4: ["No Glasses", "Glasses", "Sunglasses", "Reading Glasses"],
}


class GlassesDetector:
    """Wraps a PyTorch glasses-classification model.

    The ResNet backbone depth (18/34/50) and any wrapper prefix (e.g. ``base.``)
    are auto-detected from the checkpoint at load time, so the team's ResNet-50
    glasses model loads correctly without manual configuration.
    """

    def __init__(self, model_path: str, conf_threshold: float = _CFG_CONF,
                 device: Optional[torch.device] = None):
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf        = float(conf_threshold)
        self.model       = None
        self.num_classes = None
        self.class_names = None
        self.is_binary   = False   # True → single sigmoid output

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        self._load(str(model_path))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_model(self, backbone: str, out_dim: int) -> nn.Module:
        """Build a torchvision ResNet of the detected depth with an `out_dim` head."""
        ctor = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }.get(backbone, models.resnet18)
        net = ctor(weights=None)
        net.fc = nn.Linear(net.fc.in_features, out_dim)
        return net

    @staticmethod
    def _strip_prefix(state: dict) -> dict:
        """Strip a common wrapper prefix (the team nests the backbone under
        `base.`; other exports use backbone./model./module.) so the keys line
        up with a plain torchvision ResNet."""
        for pref in ("base.", "backbone.", "model.", "module."):
            keyed = [k for k in state if k.startswith(pref)]
            if keyed and len(keyed) >= int(0.6 * len(state)):
                return {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
        return state

    @staticmethod
    def _detect_backbone(state: dict):
        """Infer ResNet depth from the state dict. Bottleneck blocks (bn3)
        => resnet50; a 3rd block in layer4 => resnet34; else resnet18."""
        if any(k.endswith("layer1.0.bn3.weight") for k in state):
            return "resnet50"
        if "layer4.2.conv1.weight" in state:
            return "resnet34"
        return "resnet18"

    def _infer_num_classes(self, state_dict: dict) -> int:
        for key in reversed(list(state_dict.keys())):
            if "weight" in key:
                return int(state_dict[key].shape[0])
        return 2   # safe default

    def _load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            print(f"[GLASSES] Model not found at '{model_path}'. "
                  "Glasses detection DISABLED (system runs normally).")
            return
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint.to(self.device)
                for layer in reversed(list(self.model.modules())):
                    if isinstance(layer, nn.Linear):
                        out = layer.out_features
                        self.num_classes = 2 if out == 1 else out
                        self.is_binary   = (out == 1)
                        break
            else:
                state = (checkpoint.get("model_state")
                         or checkpoint.get("state_dict")
                         or checkpoint) if isinstance(checkpoint, dict) else checkpoint
                class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None

                # Normalise: strip any wrapper prefix, detect the real backbone
                # depth, and size the head from the saved fc layer. (The team's
                # glasses model is a ResNet-50 nested under `base.` with a single
                # sigmoid output — a plain resnet18 head would load nothing.)
                state = self._strip_prefix(state)
                backbone = self._detect_backbone(state)
                if "fc.weight" in state:
                    out_dim = int(state["fc.weight"].shape[0])
                else:
                    n = self._infer_num_classes(state)
                    out_dim = 1 if n <= 2 else n
                self.is_binary   = (out_dim == 1)
                self.num_classes = 2 if self.is_binary else out_dim

                self.model = self._build_model(backbone, out_dim)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(f"[GLASSES] backbone={backbone} out_dim={out_dim} "
                      f"missing={len(missing)} unexpected={len(unexpected)}")
                if len(missing) > 5:
                    print(f"[GLASSES] WARNING: {len(missing)} missing keys — "
                          "weights may be incompletely loaded.")
                self.model = self.model.to(self.device)
                if class_names:
                    self.class_names = class_names

            self.model.eval()
            if self.class_names is None:
                self.class_names = _DEFAULT_CLASS_NAMES.get(
                    self.num_classes,
                    [f"Class_{i}" for i in range(self.num_classes)],
                )
            print(f"[GLASSES] Model loaded from '{model_path}' | "
                  f"classes={self.class_names} | device={self.device}")
        except Exception as exc:
            import traceback
            print(f"[GLASSES] Failed to load model: {exc}\n{traceback.format_exc()}")
            self.model = None

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, face_crop: np.ndarray) -> Optional[str]:
        """Run inference on a BGR face crop. Returns a label or None (uncertain/disabled)."""
        if not self.is_ready or face_crop is None or face_crop.size == 0:
            return None
        try:
            rgb    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            if self.is_binary:
                prob = torch.sigmoid(logits[0, 0]).item()
                if prob < self.conf and (1 - prob) < self.conf:
                    return None
                return self.class_names[1] if prob >= self.conf else self.class_names[0]
            probs     = torch.softmax(logits[0], dim=0)
            conf, idx = probs.max(0)
            if conf.item() < self.conf:
                return None
            return self.class_names[idx.item()]
        except Exception as exc:
            print(f"[GLASSES] Inference error: {exc}")
            return None
