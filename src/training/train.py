"""
Training Script for DOTA Aerial Object Detection Models.
Trains selected candidate models using Ultralytics YOLO-OBB.
"""

import os
import sys
import time
import argparse
import logging
import yaml
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_single_model(cand_key, cand_info, cfg):
    logging.info("=" * 60)
    logging.info(f"STARTING TRAINING FOR CANDIDATE: {cand_key} ({cand_info['name']})")
    logging.info(f"Weights: {cand_info['weights']}, Description: {cand_info['description']}")
    logging.info("=" * 60)

    data_yaml = Path(cfg["dataset"]["data_yaml"]).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration YAML not found at '{data_yaml}'. Run preprocessing first!")

    # Load model
    model = YOLO(cand_info["weights"])

    train_cfg = cfg["training"]
    project_dir = Path(train_cfg["project_dir"]).resolve()
    
    start_time = time.time()

    # Train OBB model
    results = model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        workers=train_cfg["workers"],
        seed=train_cfg["seed"],
        project=str(project_dir),
        name=cand_key,
        exist_ok=True,
        verbose=False
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Training completed for {cand_key} in {elapsed_time:.2f} seconds.")

    model_save_path = project_dir / cand_key / "weights" / "best.pt"
    if not model_save_path.exists():
        model_save_path = project_dir / cand_key / "weights" / "last.pt"

    return {
        "candidate_key": cand_key,
        "name": cand_info["name"],
        "model_path": str(model_save_path),
        "duration_seconds": elapsed_time,
        "results": results
    }


def train_all_candidates(config_path="configs/config.yaml", target_model=None):
    cfg = load_config(config_path)
    candidates = cfg["candidates"]

    if target_model:
        if target_model in candidates:
            selected_candidates = {target_model: candidates[target_model]}
        else:
            # Check by candidate name (e.g. yolov8n-obb)
            matches = {k: v for k, v in candidates.items() if v["name"] == target_model}
            if matches:
                selected_candidates = matches
            else:
                logging.error(f"Target model '{target_model}' not found in candidates list!")
                sys.exit(1)
    else:
        selected_candidates = candidates

    training_summary = {}

    for cand_key, cand_info in selected_candidates.items():
        try:
            res = train_single_model(cand_key, cand_info, cfg)
            training_summary[cand_key] = res
        except Exception as e:
            logging.error(f"Error training model candidate {cand_key}: {e}", exc_info=True)

    logging.info(f"All target trainings complete. Summary count: {len(training_summary)}")
    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DOTA Object Detection Model Candidates")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Train specific candidate model (e.g., candidate_1 or yolov8n-obb)")
    args = parser.parse_args()

    train_all_candidates(config_path=args.config, target_model=args.model)
