"""
modules/embedding/embedder.py
------------------------------
Extracts high-dimensional feature vectors from cropped person images using a
lightweight Vision CNN (MobileNetV3 on PyTorch), avoiding the C++ compilation
issues found in InsightFace on Windows.
"""

import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T


class PersonEmbedder:
    """
    Extracts appearance features (embeddings) using a pretrained CNN.
    Due to InsightFace OS incompatibilities, we use MobileNetV3 (Body features).
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PersonEmbedder] Initializing MobileNetV3 on {self.device}...")
        
        # Load pre-trained model and keep only the feature extractor and pooler,
        # dropping the 1000-class ImageNet classification head.
        try:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except AttributeError:
            model = models.mobilenet_v3_small(pretrained=True)
            
        self.model = torch.nn.Sequential(
            model.features,
            model.avgpool,
            torch.nn.Flatten()  # Output shape: (1, 576)
        )
        self.model.to(self.device).eval()
        
        # Standard ImageNet pre-processing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[PersonEmbedder] Embedder ready (Body / MobileNetV3).")

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
        Extract a 576-dimensional normalized body feature vector.
        
        Args:
            crop (np.ndarray): BGR image patch.
            
        Returns:
            np.ndarray: 1D normalized feature vector (float32).
        """
        if crop.size == 0:
            # Fallback if an empty crop is somehow passed
            return np.zeros(576, dtype=np.float32)

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
