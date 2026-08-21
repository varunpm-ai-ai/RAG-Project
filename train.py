"""
Root Entry Point for Training DOTA Object Detection Model Candidates.
Usage:
    python train.py                   # Train all candidates
    python train.py --model candidate_1 # Train specific candidate
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from src.training.train import train_all_candidates
from src.training.evaluate import evaluate_all_candidates
from src.training.select_best import select_and_export_best_model


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate DOTA Model Candidates")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Train specific candidate model")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation + best model selection")
    args = parser.parse_args()

    if not args.eval_only:
        print("Starting candidate training...")
        train_all_candidates(config_path=args.config, target_model=args.model)

    print("Evaluating trained candidate models...")
    evaluate_all_candidates(config_path=args.config)

    print("Selecting top model and saving artifact...")
    select_and_export_best_model(config_path=args.config)


if __name__ == "__main__":
    main()
