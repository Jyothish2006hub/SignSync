"""Step 4a: Run MediaPipe Hands over the captured dataset and save landmark features.

Usage:
    python extract_landmarks.py --dataset dataset --out landmarks.csv
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from common import FEATURE_COLUMNS, landmarks_to_row

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Dataset directory (one subfolder per letter)")
    parser.add_argument("--out", default="landmarks.csv", help="Output CSV path")
    args = parser.parse_args()

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    rows_written = 0
    skipped = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_COLUMNS + ["label"])

        for label in sorted(os.listdir(args.dataset)):
            label_dir = os.path.join(args.dataset, label)
            if not os.path.isdir(label_dir):
                continue

            for filename in sorted(os.listdir(label_dir)):
                path = os.path.join(label_dir, filename)
                image = cv2.imread(path)
                if image is None:
                    skipped += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    skipped += 1
                    continue

                row = landmarks_to_row(result.hand_landmarks)
                writer.writerow(row + [label])
                rows_written += 1

    landmarker.close()
    print(f"Wrote {rows_written} rows to {args.out} ({skipped} images skipped — no hand detected).")


if __name__ == "__main__":
    main()
