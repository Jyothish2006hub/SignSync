"""SignSync: real-time sign language translation from the laptop webcam.

Usage:
    python realtime_infer.py
    python realtime_infer.py --webcam-index 1 --model model.joblib

Controls (in the preview window):
    c       clear the displayed text
    q       quit
"""

import argparse
import os
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from common import FEATURE_COLUMNS, landmarks_to_row

MODEL_TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Sliding window majority vote: a sign confirms once it holds this fraction
# of the last WINDOW_SIZE frames, tolerating brief flicker/tremor instead of
# requiring strictly consecutive identical predictions.
WINDOW_SIZE = 6
CONFIRM_RATIO = 0.65

# (start, end) point-index pairs for drawing the hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--model", default="model.joblib", help="Trained classifier from train_classifier.py")
    args = parser.parse_args()

    model = joblib.load(args.model)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    frame_timestamp_ms = 0

    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.webcam_index}")

    displayed_text = ""
    confirmed_prediction = None
    recent_predictions = deque(maxlen=WINDOW_SIZE)

    print("Press c=clear, q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp_ms += 33  # assume ~30fps; only needs to be monotonically increasing
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        prediction = None
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
            row = pd.DataFrame([landmarks_to_row(result.hand_landmarks)], columns=FEATURE_COLUMNS)
            prediction = model.predict(row)[0]

        recent_predictions.append(prediction)
        counts = Counter(p for p in recent_predictions if p is not None)
        top_prediction, top_count = (counts.most_common(1)[0] if counts else (None, 0))
        is_stable = (
            len(recent_predictions) == WINDOW_SIZE
            and top_prediction is not None
            and top_count / WINDOW_SIZE >= CONFIRM_RATIO
        )

        if is_stable and top_prediction != confirmed_prediction:
            confirmed_prediction = top_prediction
            displayed_text = top_prediction
            print(f"Confirmed: {displayed_text}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            displayed_text = ""
            confirmed_prediction = None

        cv2.putText(
            frame,
            f"pred: {prediction}  showing: {displayed_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("SignSync", frame)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
