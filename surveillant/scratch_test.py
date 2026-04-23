import cv2
import numpy as np
from modules.embedding.embedder import PersonEmbedder

v1 = cv2.imread("data/snapshots/179f1cd4-8b05-4b45-aa42-ec5b037bb4ed/crop_0.jpg")
v2 = cv2.imread("data/snapshots/179f1cd4-8b05-4b45-aa42-ec5b037bb4ed/crop_1.jpg")

emb = PersonEmbedder()
f1 = emb.extract_body_embedding(v1)
f2 = emb.extract_body_embedding(v2)
print("Same guy two frames sim:", np.dot(f1, f2))

v3 = np.zeros((100,200,3), dtype=np.uint8)
f3 = emb.extract_body_embedding(v3)
print("Guy vs Black image:", np.dot(f1, f3))
