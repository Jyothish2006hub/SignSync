# SignSync: Sign Language Translator

SignSync translates hand signs into text in real time, using a laptop webcam and screen. MediaPipe finds the hand landmarks in each frame, and a Random Forest classifier trained on those landmarks recognises the sign.

## Supported signs

| Key | Sign |
|-----|------|
| 1 | yes |
| 2 | okay |
| 3 | i didnt understand |
| 4 | perfect |
| 5 | help |
| 6 | i need food |
| 7 | hi how are you |
| 8 | WHATS YOUR NAME |
| 9 | CALL MY FAMILY |
| 0 | where is restroom |

The number keys are the shortcuts used by `capture_dataset.py`.

## Run the translator

Requires Python 3.10–3.12.

```
pip install -r requirements.txt
python realtime_infer.py
```

Hold a sign steady in front of the webcam. The window draws the hand skeleton and shows the recognised sign once it has held for most of the last few frames. Press `c` to clear the text and `q` to quit.

## Retrain on your own signs

1. **Capture images.** Use `python capture_dataset.py` to save frames one at a time with the space bar. Use `python auto_capture.py --label "help" --count 200 --require-hands 1` to save them automatically. Images go to `dataset/<sign>/`.
2. **Extract landmarks.** Run `python extract_landmarks.py --dataset dataset --out landmarks.csv`.
3. **Train.** Run `python train_classifier.py --csv landmarks.csv --out model.joblib`.

## Files

| File | Purpose |
|------|---------|
| `realtime_infer.py` | Live webcam translator |
| `capture_dataset.py`, `auto_capture.py` | Dataset capture from the webcam |
| `extract_landmarks.py` | Dataset images → hand-landmark CSV |
| `train_classifier.py` | Trains the classifier (with jitter augmentation) |
| `common.py` | Shared feature layout (2 hands × 21 landmarks × xyz) |
| `hand_landmarker.task` | MediaPipe hand landmark model |
| `model.joblib` | Trained classifier for the 10 signs |
| `landmarks.csv` | Extracted landmarks for the included dataset |
| `dataset/` | About 500 webcam images per sign |
