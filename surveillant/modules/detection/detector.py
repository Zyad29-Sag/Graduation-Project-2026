"""
modules/detection/detector.py
------------------------------
PersonDetector wraps YOLOv8 to detect only people (class 0) in a frame.

The YOLOv8 model is downloaded automatically on first use by `ultralytics`.
"""

import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO


class PersonDetector:
    """
    Detects people in a single video frame using YOLOv8.

    The model is loaded once at construction time and reused for every
    subsequent call to `detect()`.

    Args:
        model_name (str):  YOLOv8 model filename, e.g. ``"yolov8n.pt"``.
        conf       (float): Minimum detection confidence (0–1).
        imgsz      (int):  Input image size passed to YOLO. Smaller = faster.
                           320 is ~3× faster than 640 for CPU inference.
    """

    PERSON_CLASS_ID = 0  # class 0 = person in the COCO dataset

    def __init__(self, model_name: str, conf: float, imgsz: int = 320) -> None:
        self.model_name: str   = model_name
        self.conf:       float = conf
        self.imgsz:      int   = imgsz

        print(f"[PersonDetector] Loading model: {model_name} (imgsz={imgsz}) …")
        self._model = YOLO(model_name)
        print(f"[PersonDetector] Model ready (conf≥{conf}, imgsz={imgsz}).")

    def __repr__(self) -> str:
        return (
            f"PersonDetector(model={self.model_name!r}, "
            f"conf={self.conf}, imgsz={self.imgsz})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run person detection on a single frame.

        Args:
            frame (np.ndarray): BGR image as returned by OpenCV.

        Returns:
            List of dicts, each with keys:
                - ``bbox``       : ``[x1, y1, x2, y2]`` in pixel coordinates.
                - ``confidence`` : float between 0 and 1.

            Returns an empty list if no people are detected.
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            classes=[self.PERSON_CLASS_ID],
            imgsz=self.imgsz,
            iou=0.40,    # aggressive NMS — prevents split detections (torso + full body) reaching the tracker
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                # Skip tiny boxes — likely false positives or body fragments
                if w < 20 or h < 40:
                    continue
                detections.append(
                    {
                        "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf[0]),
                    }
                )

        return detections
