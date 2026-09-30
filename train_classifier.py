"""Step 4b: Train a letter classifier on the extracted hand-landmark features.

Usage:
    python train_classifier.py --csv landmarks.csv --out model.joblib

Training data is augmented with small random jitter on each landmark
coordinate to simulate natural hand tremor / camera noise, so the model
doesn't flicker between classes when a live hand isn't perfectly still.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from common import FEATURE_COLUMNS

# Number of jittered copies generated per training sample (0 = no augmentation).
JITTER_COPIES = 4
# Std-dev of the per-coordinate Gaussian noise, in the same normalized units
# MediaPipe uses for x/y/z (roughly fraction of frame width/height).
JITTER_STD = 0.01


def augment_with_jitter(X, y, copies, std, seed=42):
    rng = np.random.default_rng(seed)
    X_arr = X.to_numpy()
    # Zero-padded (missing-hand) columns stay exactly zero — jittering them
    # would fabricate a second hand that was never there.
    nonzero_mask = X_arr != 0

    X_aug = [X_arr]
    y_aug = [y.to_numpy()]
    for _ in range(copies):
        noise = rng.normal(0.0, std, size=X_arr.shape) * nonzero_mask
        X_aug.append(X_arr + noise)
        y_aug.append(y.to_numpy())

    X_combined = pd.DataFrame(np.vstack(X_aug), columns=X.columns)
    y_combined = pd.Series(np.concatenate(y_aug))
    return X_combined, y_combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="landmarks.csv", help="Landmarks CSV from extract_landmarks.py")
    parser.add_argument("--out", default="model.joblib", help="Where to save the trained model")
    parser.add_argument("--jitter-copies", type=int, default=JITTER_COPIES, help="Jittered copies per training sample (0 disables augmentation)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.jitter_copies > 0:
        X_train, y_train = augment_with_jitter(X_train, y_train, args.jitter_copies, JITTER_STD)

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
