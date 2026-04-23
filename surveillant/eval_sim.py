import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.detection.detector import PersonDetector
from modules.embedding.embedder import PersonEmbedder
from config.settings import YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ

detector = PersonDetector(YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ)
embedder = PersonEmbedder()

cap1 = cv2.VideoCapture("data/videos/video1_1.avi")
cap2 = cv2.VideoCapture("data/videos/video1_3.avi") # different camera

# grab first non-empty detection from cam 1
feat1 = None
for _ in range(30):
    ret, frame = cap1.read()
    det = detector.detect(frame)
    if det:
        x1, y1, x2, y2 = det[0]['bbox']
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        feat1 = embedder.extract_body_embedding(crop)
        break

# grab first non-empty detection from cam 3
feat2 = None
for _ in range(30):
    ret, frame = cap2.read()
    det = detector.detect(frame)
    if det:
        x1, y1, x2, y2 = det[0]['bbox']
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        feat2 = embedder.extract_body_embedding(crop)
        break

if feat1 is not None and feat2 is not None:
    score = cosine_similarity(feat1.reshape(1,-1), feat2.reshape(1,-1))[0][0]
    print(f"SCORE BETWEEN TWO RANDOM PEOPLE ON DIFFERENT CAMERAS: {score:.4f}")

cap1.release()
cap2.release()
