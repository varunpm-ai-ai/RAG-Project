"""
Evaluation Script for Trained DOTA Object Detection Models.
Evaluates model candidates on test/val set, captures Precision, Recall, mAP50, mAP50-95,
Inference Speed, Model Size, and Per-Class Metrics, saving comparison CSV/JSON.
"""

import os
import sys
import time
import json
import yaml
import logging
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO
from src.utils.constants import DOTA_CLASSES, ID_TO_CLASS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model(model_path, data_yaml, split="test", imgsz=512, device="cpu"):
    logging.info(f"Evaluating model weights: {model_path} on split: {split}")
    
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model weight file '{model_path}' does not exist!")

    model_size_mb = path_obj.stat().st_size / (1024 * 1024)

    model = YOLO(str(path_obj))

    start_t = time.time()
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False
    )
    total_val_time = time.time() - start_t

    # Extract metrics robustly
    try:
        results_dict = metrics.results_dict
        precision = float(results_dict.get("metrics/precision(B)", 0.0))
        recall = float(results_dict.get("metrics/recall(B)", 0.0))
        map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
    except Exception:
        # Fallback metric extraction
        precision = float(getattr(metrics.box, "mp", 0.0)) if hasattr(metrics, "box") else 0.0
        recall = float(getattr(metrics.box, "mr", 0.0)) if hasattr(metrics, "box") else 0.0
        map50 = float(getattr(metrics.box, "map50", 0.0)) if hasattr(metrics, "box") else 0.0
        map50_95 = float(getattr(metrics.box, "map", 0.0)) if hasattr(metrics, "box") else 0.0

    # Timing metrics
    speed_dict = getattr(metrics, "speed", {})
    inference_time_ms = float(speed_dict.get("inference", 0.0))

    # Per-class metrics
    per_class_metrics = {}
    try:
        class_maps = metrics.box.maps if hasattr(metrics, "box") and hasattr(metrics.box, "maps") else []
        for cls_id, cls_name in enumerate(DOTA_CLASSES):
            m50_95 = float(class_maps[cls_id]) if cls_id < len(class_maps) else 0.0
            per_class_metrics[cls_name] = round(m50_95, 4)
    except Exception as e:
        logging.warning(f"Could not extract per-class maps: {e}")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "model_size_mb": round(model_size_mb, 2),
        "inference_speed_ms": round(inference_time_ms, 2),
        "per_class_mAP50_95": per_class_metrics
    }


def evaluate_all_candidates(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    candidates = cfg["candidates"]
    data_yaml = Path(cfg["dataset"]["data_yaml"]).resolve()
    project_dir = Path(cfg["training"]["project_dir"]).resolve()
    reports_dir = Path(cfg["artifacts"]["reports_dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_list = []

    for cand_key, cand_info in candidates.items():
        cand_dir = project_dir / cand_key
        best_weights = cand_dir / "weights" / "best.pt"
        if not best_weights.exists():
            best_weights = cand_dir / "weights" / "last.pt"

        if not best_weights.exists():
            logging.warning(f"No trained weights found for candidate '{cand_key}' in '{cand_dir}'. Skipping evaluation.")
            continue

        try:
            eval_res = evaluate_model(
                model_path=str(best_weights),
                data_yaml=str(data_yaml),
                split="val",
                imgsz=cfg["training"]["imgsz"],
                device=cfg["training"]["device"]
            )

            record = {
                "candidate_key": cand_key,
                "model_name": cand_info["name"],
                "description": cand_info["description"],
                "weights_path": str(best_weights),
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "mAP50": eval_res["mAP50"],
                "mAP50_95": eval_res["mAP50_95"],
                "size_mb": eval_res["model_size_mb"],
                "inference_speed_ms": eval_res["inference_speed_ms"],
                "per_class_metrics": eval_res["per_class_mAP50_95"]
            }
            summary_list.append(record)
        except Exception as e:
            logging.error(f"Error evaluating candidate {cand_key}: {e}", exc_info=True)

    if not summary_list:
        logging.error("No valid candidate evaluations were completed!")
        return None

    csv_path = reports_dir / "model_comparison.csv"
    json_path = reports_dir / "model_comparison.json"

    fieldnames = ["Candidate", "Model Name", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Size (MB)", "Inference Speed (ms)"]
    rows = [
        {
            "Candidate": item["candidate_key"],
            "Model Name": item["model_name"],
            "Precision": item["precision"],
            "Recall": item["recall"],
            "mAP@0.5": item["mAP50"],
            "mAP@0.5:0.95": item["mAP50_95"],
            "Size (MB)": item["size_mb"],
            "Inference Speed (ms)": item["inference_speed_ms"]
        }
        for item in summary_list
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    header_str = f"{'Candidate':<13} {'Model Name':<15} {'Precision':<10} {'Recall':<8} {'mAP50':<8} {'mAP50-95':<10} {'Size(MB)':<10} {'Speed(ms)':<10}"
    print(header_str)
    print("-" * len(header_str))
    for r in rows:
        print(f"{r['Candidate']:<13} {r['Model Name']:<15} {r['Precision']:<10.4f} {r['Recall']:<8.4f} {r['mAP@0.5']:<8.4f} {r['mAP@0.5:0.95']:<10.4f} {r['Size (MB)']:<10.2f} {r['Inference Speed (ms)']:<10.2f}")
    print("=" * 70)
    print(f"Reports saved to '{csv_path}' and '{json_path}'\n")

    return summary_list


if __name__ == "__main__":
    evaluate_all_candidates()
