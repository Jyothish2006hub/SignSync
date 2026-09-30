"""Fast auto-capture: continuously save webcam frames under one label, no keypresses needed.

Usage:
    python auto_capture.py --label perfect --count 200
    python auto_capture.py --label "WHATS YOUR NAME" --count 200 --require-hands 2

Hold the gesture steady in front of the camera; frames save automatically every
~100ms. With --require-hands, a frame is only saved if MediaPipe actually detects
that many hands in it (filters out bad/partial captures at the source instead of
after the fact). Press q to stop early.
"""

import argparse
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label/class name for this capture run")
    parser.add_argument("--count", type=int, default=200, help="Number of frames to capture")
    parser.add_argument("--out", default="dataset", help="Output dataset directory")
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between saved frames")
    parser.add_argument(
        "--require-hands",
        type=int,
        default=0,
        help="Only save a frame if exactly this many hands are detected (0 = no check)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.webcam_index}")

    landmarker = None
    frame_ts = 0
    if args.require_hands > 0:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max(2, args.require_hands),
            min_hand_detection_confidence=0.6,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)

    label_dir = os.path.join(args.out, args.label)
    os.makedirs(label_dir, exist_ok=True)
    existing = len(os.listdir(label_dir))

    print(f"Capturing {args.count} frames for label '{args.label}'. Hold the gesture steady.")
    if args.require_hands:
        print(f"Quality gate: only saving frames with exactly {args.require_hands} hand(s) detected.")
    print("Starting in 3 seconds...")
    time.sleep(3)

    saved = 0
    rejected = 0
    last_save = 0.0
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            continue

        num_detected = None
        if landmarker is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            frame_ts += 33
            result = landmarker.detect_for_video(mp_image, frame_ts)
            num_detected = len(result.hand_landmarks)

        display_frame = frame.copy()
        status = f"label: {args.label}  saved: {saved}/{args.count}"
        if num_detected is not None:
            status += f"  hands: {num_detected}/{args.require_hands}"
        cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("auto_capture", display_frame)

        now = time.time()
        quality_ok = num_detected is None or num_detected == args.require_hands
        if now - last_save >= args.interval:
            if quality_ok:
                idx = existing + saved
                path = os.path.join(label_dir, f"{idx:04d}.jpg")
                cv2.imwrite(path, frame)
                saved += 1
                last_save = now
                print(f"Saved {saved}/{args.count}: {path}")
            else:
                rejected += 1
                last_save = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if landmarker is not None:
        landmarker.close()
    print(f"Done. Captured {saved} frames for '{args.label}' ({rejected} rejected for wrong hand count).")


if __name__ == "__main__":
    main()
