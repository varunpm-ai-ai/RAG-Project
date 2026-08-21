"""
CLI Inference Script for DOTA Object Detection.
Usage:
    python inference.py --image path/to/aerial_image.jpg --conf 0.25 --output reports/annotated_output.jpg
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from src.inference.predictor import DotaPredictor


def main():
    parser = argparse.ArgumentParser(description="DOTA Aerial Object Detection Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="artifacts/best_model/model.pt", help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="reports/annotated_output.jpg", help="Path to save annotated image")
    args = parser.parse_args()

    predictor = DotaPredictor(model_path=args.model)
    res = predictor.predict(image_input=args.image, conf_threshold=args.conf)

    print("\n" + "=" * 50)
    print(f"INFERENCE RESULTS FOR: {args.image}")
    print("=" * 50)

    detections = res["detections"]
    if not detections:
        print("No objects detected above confidence threshold.")
    else:
        print(f"Total Objects Detected: {len(detections)}")
        print("\nDetected Objects Breakdown:")
        for idx, det in enumerate(detections, 1):
            print(f"  {idx}. Category: {det['class_name']:20s} | Confidence: {det['confidence']*100:.1f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res["annotated_image"].save(out_path)

    print("=" * 50)
    print(f"Annotated image saved to: '{out_path.resolve()}'\n")


if __name__ == "__main__":
    main()
