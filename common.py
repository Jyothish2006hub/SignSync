"""Shared helpers for the sign-language recognition pipeline."""

import string

# ASL alphabet target classes (static hand-shape letters only).
LETTERS = list(string.ascii_uppercase)

# 21 hand landmarks x (x, y, z) = 63 features, in MediaPipe's fixed point order.
NUM_LANDMARKS = 21
MAX_HANDS = 2

FEATURE_COLUMNS = [
    f"h{h}_lm{i}_{axis}"
    for h in range(MAX_HANDS)
    for i in range(NUM_LANDMARKS)
    for axis in ("x", "y", "z")
]


def landmarks_to_row(hands_landmarks):
    """Flatten up to MAX_HANDS lists of 21 MediaPipe landmark points into a fixed
    126-value feature vector (63 per hand). Missing hands are zero-padded.
    Hands are ordered left-to-right by wrist x-position so the feature layout
    is consistent regardless of MediaPipe's detection order.
    """
    if hands_landmarks and hasattr(hands_landmarks[0], "x"):
        # Backward-compat: called with a single hand's landmark list directly.
        hands_landmarks = [hands_landmarks]

    hands_sorted = sorted(hands_landmarks, key=lambda lm_list: lm_list[0].x)[:MAX_HANDS]

    row = []
    for hand_landmarks in hands_sorted:
        for lm in hand_landmarks:
            row.extend([lm.x, lm.y, lm.z])
    while len(hands_sorted) < MAX_HANDS:
        row.extend([0.0] * (NUM_LANDMARKS * 3))
        hands_sorted.append(None)
    return row
