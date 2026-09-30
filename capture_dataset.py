"""Capture a hand-sign image dataset from the laptop webcam, one keypress per frame.

Usage:
    python capture_dataset.py
    python capture_dataset.py --webcam-index 1

Controls (in the preview window):
    0-9     set the current label to a sign from CUSTOM_LABELS below
    space   save the current frame under the current label
    q       quit

Saved images land in dataset/<LABEL>/<NNNN>.jpg
"""

import argparse
import os

import cv2

# Number-key shortcuts for the sign labels (edit this list as needed).
CUSTOM_LABELS = {
    ord("1"): "yes",
    ord("2"): "okay",
    ord("3"): "i didnt understand",
    ord("4"): "perfect",
    ord("5"): "help",
    ord("6"): "i need food",
    ord("7"): "hi how are you",
    ord("8"): "WHATS YOUR NAME",
    ord("9"): "CALL MY FAMILY",
    ord("0"): "where is restroom",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--out", default="dataset", help="Output dataset directory")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.webcam_index}")

    current_label = CUSTOM_LABELS[ord("1")]
    counts = {}

    print("Press 0-9 to pick a sign, space to save a frame, q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        label_dir = os.path.join(args.out, current_label)
        if current_label not in counts:
            # Continue numbering after any images already saved for this label.
            counts[current_label] = len(os.listdir(label_dir)) if os.path.isdir(label_dir) else 0

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"label: {current_label}  saved: {counts[current_label]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("capture_dataset", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in CUSTOM_LABELS:
            current_label = CUSTOM_LABELS[key]
            print(f"Label set to: {current_label}")
        elif key == ord(" "):
            os.makedirs(label_dir, exist_ok=True)
            idx = counts[current_label]
            path = os.path.join(label_dir, f"{idx:04d}.jpg")
            cv2.imwrite(path, frame)
            counts[current_label] = idx + 1
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
