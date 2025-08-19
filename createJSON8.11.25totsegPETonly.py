#!/usr/bin/env python3
import json
from pathlib import Path

# --- Fixed locations (edit if your folders move) ---
BASE = "/media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3/FDG"
PET_DIR    = f"{BASE}/PET_rescaled_bodyCrop"
TTB_DIR    = f"{BASE}/TTB_phys_bodyCrop"

# --- Build 5 folds over 100 cases (0001..0100) round-robin ---
training = []
for i in range(1, 101):
    case = f"{i:04d}"
    fold = (i - 1) % 5  # folds 0..4; first 4 folds will have 20, last 1 have 19

    entry = {
        "fold": fold,
        "image": [
            f"{PET_DIR}/train_{case}.nii.gz",
        ],
        "label": f"{TTB_DIR}/train_{case}.nii.gz",
    }
    training.append(entry)

data = {
    "training": training,
    "testing": []  # none held out
}

# Save JSON right where you run this script
out_path = Path.cwd() / "FDG_5fold_train.json"
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

# Optional: small summary printed to terminal
counts = {k: 0 for k in range(5)}
for e in training:
    counts[e["fold"]] += 1
print(f"Wrote {out_path}")
print("Per-fold counts:", counts)
