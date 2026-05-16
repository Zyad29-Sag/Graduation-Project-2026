"""
modules/embedding/embedder.py
------------------------------
Extracts high-dimensional feature vectors from cropped person images using
ResNet-50 (PyTorch). ResNet-50 provides 2048-d features that discriminate
appearance far better than MobileNetV3-small for Re-ID tasks.

NOTE: Switching backbones invalidates previously stored embeddings.
      Clear the database (surveillant.db) when changing this model.
"""

import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T

EMBEDDING_DIM = 2048  # ResNet-50 pooled feature size


class PersonEmbedder:
    """
    Extracts appearance features (embeddings) using ResNet-50.
    The classification head is removed; only the 2048-d pooled feature vector
    is used so that cosine similarity measures appearance, not ImageNet class.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PersonEmbedder] Initializing ResNet-50 on {self.device}...")

        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            backbone = models.resnet50(pretrained=True)

        # Strip the final FC classification layer; keep conv + avgpool → 2048-d
        self.model = torch.nn.Sequential(
            *list(backbone.children())[:-1],  # up to and including avgpool
            torch.nn.Flatten(),               # (batch, 2048, 1, 1) → (batch, 2048)
        )
        self.model.to(self.device).eval()

        # Standard ImageNet pre-processing (ResNet expects 224×224)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[PersonEmbedder] Embedder ready (Body / ResNet-50, dim=2048).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_face_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Face extraction is bypassed in this setup.
        Returns None to automatically trigger body fallback.
        """
        return None

    def extract_body_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a 2048-dimensional normalized body feature vector.

        Args:
            crop (np.ndarray): BGR image patch.

        Returns:
            np.ndarray: 1D normalized feature vector (float32).
        """
        if crop.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(tensor).cpu().numpy().flatten()
            
        # L2 Normalize the vector so cosine similarity works via simple dot product
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
            
        return feat.astype(np.float32)

    def aggregate_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Average a list of embeddings and re-normalize.
        
        Args:
            embeddings: List of 1D numpy arrays.
            
        Returns:
            np.ndarray: A single normalized 1D numpy array representing the prototype.
        """
        if not embeddings:
            raise ValueError("No embeddings provided to aggregate.")
            
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
            
        return avg.astype(np.float32)

    def serialize(self, embedding: np.ndarray) -> bytes:
        """Convert a numpy array to raw bytes for SQLite BLOB storage."""
        return embedding.tobytes()

    def deserialize(self, data: bytes) -> np.ndarray:
        """Convert raw bytes back into a numpy float32 array."""
        return np.frombuffer(data, dtype=np.float32)
