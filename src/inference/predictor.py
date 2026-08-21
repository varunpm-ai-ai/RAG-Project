"""
Reusable Predictor Module for DOTA Aerial Object Detection.
Loads trained best model artifact and performs inference on images.
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO
from src.utils.constants import ID_TO_CLASS, DOTA_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DotaPredictor:
    def __init__(self, model_path="artifacts/best_model/model.pt", metadata_path="artifacts/best_model/metadata.json"):
        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'. "
                "Please run training and select_best.py first!"
            )

        logging.info(f"Loading object detection model from: '{self.model_path}'")
        self.model = YOLO(str(self.model_path))

        self.class_mapping = ID_TO_CLASS
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if "class_mapping" in meta:
                        self.class_mapping = {int(k): v for k, v in meta["class_mapping"].items()}
            except Exception as e:
                logging.warning(f"Could not read class mapping from metadata: {e}")

    def predict(self, image_input, conf_threshold=0.25, iou_threshold=0.45):
        """
        Runs object detection on image input (file path, PIL Image, or numpy array).

        Returns dict:
        {
            "detections": [
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "box": list (coordinates)
                }, ...
            ],
            "annotated_image": PIL Image,
            "detected_classes": list of unique detected class names
        }
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image '{img_path}' does not exist!")
            image_src = str(img_path)
        else:
            image_src = image_input

        results = self.model.predict(
            source=image_src,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        res = results[0]
        detections = []

        # Check for OBB detections or standard BBOX detections
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            boxes = res.obb.xyxyxyxy.cpu().numpy() if hasattr(res.obb, "xyxyxyxy") else res.obb.xywhr.cpu().numpy()
            confs = res.obb.conf.cpu().numpy()
            clss = res.obb.cls.cpu().numpy()

            for i in range(len(clss)):
                cls_id = int(clss[i])
                cls_name = self.class_mapping.get(cls_id, DOTA_CLASSES[cls_id] if cls_id < len(DOTA_CLASSES) else f"cls_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(confs[i]),
                    "box": boxes[i].tolist()
                })

        elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()

            for i in range(len(clss)):
                cls_id = int(clss[i])
                cls_name = self.class_mapping.get(cls_id, DOTA_CLASSES[cls_id] if cls_id < len(DOTA_CLASSES) else f"cls_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(confs[i]),
                    "box": boxes[i].tolist()
                })

        # Render annotated image
        annotated_bgr = res.plot()
        annotated_rgb = annotated_bgr[..., ::-1]  # Convert BGR to RGB
        annotated_pil = Image.fromarray(annotated_rgb)

        detected_classes = list(set([d["class_name"] for d in detections]))

        return {
            "detections": detections,
            "detected_classes": detected_classes,
            "annotated_image": annotated_pil,
            "raw_result": res
        }


if __name__ == "__main__":
    predictor = DotaPredictor()
    print("DotaPredictor initialized successfully.")
