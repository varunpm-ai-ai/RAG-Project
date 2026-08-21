"""
Best Model Selection and Artifact Export Utility.
Identifies the top-performing model candidate based on mAP@0.5:0.95
and exports the model weight artifact and metadata for Streamlit integration.
"""

import os
import sys
import json
import shutil
import yaml
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
import ultralytics
from src.utils.constants import DOTA_CLASSES, ID_TO_CLASS
from src.training.evaluate import evaluate_all_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_and_export_best_model(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["artifacts"]["reports_dir"]).resolve()
    json_path = reports_dir / "model_comparison.json"

    # If reports don't exist, run evaluation first
    if not json_path.exists():
        logging.info("Model comparison json not found. Running evaluation for all candidates...")
        eval_summary = evaluate_all_candidates(config_path)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)

    if not eval_summary:
        logging.error("No model candidates available for selection!")
        return None

    # Rank by mAP50_95 primary, mAP50 secondary
    best_candidate = max(
        eval_summary,
        key=lambda x: (x.get("mAP50_95", 0.0), x.get("mAP50", 0.0), -x.get("size_mb", 999.0))
    )

    logging.info("=" * 60)
    logging.info(f"SELECTED BEST MODEL: {best_candidate['candidate_key']} ({best_candidate['model_name']})")
    logging.info(f"mAP@0.5:0.95: {best_candidate['mAP50_95']}, mAP@0.5: {best_candidate['mAP50']}")
    logging.info("=" * 60)

    # Destination directory: artifacts/best_model
    artifacts_dir = Path(cfg["artifacts"]["best_model_dir"]).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    src_weights = Path(best_candidate["weights_path"])
    dst_weights = artifacts_dir / "model.pt"

    shutil.copy2(src_weights, dst_weights)
    logging.info(f"Copied model weights to '{dst_weights}'")

    # Export class_names.json
    class_names_path = artifacts_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(ID_TO_CLASS, f, indent=2)
    logging.info(f"Exported class mapping to '{class_names_path}'")

    # Export metadata.json
    metadata = {
        "selected_at": datetime.now().isoformat(),
        "candidate_key": best_candidate["candidate_key"],
        "model_name": best_candidate["model_name"],
        "description": best_candidate.get("description", ""),
        "training_config": cfg["training"],
        "dataset_config": cfg["dataset"],
        "metrics": {
            "precision": best_candidate["precision"],
            "recall": best_candidate["recall"],
            "mAP50": best_candidate["mAP50"],
            "mAP50_95": best_candidate["mAP50_95"],
            "model_size_mb": best_candidate["size_mb"],
            "inference_speed_ms": best_candidate["inference_speed_ms"]
        },
        "per_class_metrics": best_candidate.get("per_class_metrics", {}),
        "class_mapping": ID_TO_CLASS,
        "environment": {
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }

    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Exported artifact metadata to '{metadata_path}'")
    print("\nBEST MODEL ARTIFACT CREATED SUCCESSFULLY!")
    print(f"Artifact Location: {artifacts_dir}\n")

    return metadata


if __name__ == "__main__":
    select_and_export_best_model()
