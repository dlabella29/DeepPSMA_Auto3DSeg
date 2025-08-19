#!/usr/bin/env python3
"""
PSMA instance-wise lesion size analysis with 18-connected components (no flags).

Units: **voxels** (not mL). Histograms are plotted from 0 to 500 voxels with 10-voxel bins.

Fixed input paths inside the script:
  - TTB/Phys masks (labels: 1=TTB, 2=Physiologic):
      /media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3/PSMA/TTB_phys_bodyCrop
  - Totalseg masks (labels: 1-3=bone, 4=other organs):
      /media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3/PSMA/totseg_resampled_bodyCrop

Output directory (created in current working directory):
  - DeepPSMAinstanceLesionAnalysis

Produces:
  - lesion_instance_stats.csv         : per-lesion metrics (voxel_count & overlaps)
      * overlap_bone_voxels (totseg 1-3 summed)
      * overlap_other_organs_voxels (totseg == 4, remapped "other organs")
      * overlap_other_organs_original_total_voxels (sum of original organ labels: 4,5,6,7,8,9,15,17,18,19,20,21)
      * per-original-totseg-label overlap columns for labels:
        1,2,3,4,5,6,7,8,9,15,17,18,19,20,21  (columns named overlap_totseg_orig_<label>_voxels)
  - hist_ttb_voxels.png               : histogram of TTB lesion sizes (voxels)
  - hist_physiologic_voxels.png       : histogram of physiologic lesion sizes (voxels)
  - voxel_summary_stats.csv           : mean/median/p10/p90 of instance sizes in voxels

Run:
  python psma_instance_volume_analysis_noflags.py

Dependencies:
  pip install nibabel numpy scipy matplotlib pandas
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel import processing as niproc
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Fixed configuration (paths)
# ---------------------------
TTB_DIR = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3/PSMA/TTB_phys_bodyCrop")
TOTSEG_DIR = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3/PSMA/totseg_resampled_bodyCrop")
OUTPUT_DIR = Path.cwd() / "DeepPSMAinstanceLesionAnalysis"
FILE_GLOB = "*.nii*"  # matches .nii and .nii.gz

# ---------------------------
# Histogram configuration (voxels)
# ---------------------------
HIST_X_MIN = 0
HIST_X_MAX = 500
HIST_BIN_STEP = 10  # voxels
HIST_BINS = np.arange(HIST_X_MIN, HIST_X_MAX + HIST_BIN_STEP, HIST_BIN_STEP)  # inclusive 500

# ---------------------------
# Totseg label configuration
# ---------------------------
# Original totseg labels to report overlap for (prior to any remapping)
TOTSEG_ORIGINAL_LABELS_TO_REPORT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 17, 18, 19, 20, 21]
# For convenience: define which are "bone" and which are "other organs (original)"
TOTSEG_BONE_LABELS = [1, 2, 3]
TOTSEG_OTHER_ORG_ORIGINAL_LABELS = [4, 5, 6, 7, 8, 9, 15, 17, 18, 19, 20, 21]


def generate_18_structure() -> np.ndarray:
    """Return a 3D structuring element for 18-connectivity (faces + edges)."""
    return ndimage.generate_binary_structure(rank=3, connectivity=2)


def find_matching_totseg(ttb_path: Path, totseg_dir: Path) -> Optional[Path]:
    """
    Find a totseg file in `totseg_dir` with the same base name as `ttb_path`,
    allowing for either .nii or .nii.gz extensions.
    """
    base = ttb_path.name
    base_stem = base.replace(".nii.gz", "").replace(".nii", "")
    # Try exact same filename first
    direct = totseg_dir / base
    if direct.exists():
        return direct
    # Try matching by stem
    candidates = list(totseg_dir.glob(f"{base_stem}.nii*"))
    if len(candidates) > 0:
        # Prefer .nii.gz if both exist
        candidates_sorted = sorted(candidates, key=lambda p: (p.suffix != ".gz", str(p)))
        return candidates_sorted[0]
    return None


def load_nifti(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI image from `path`."""
    return nib.load(str(path))


def ensure_same_grid(moving_img: nib.Nifti1Image, reference_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Ensure `moving_img` is resampled onto the grid of `reference_img` (shape + affine).
    Uses nearest-neighbor (order=0) because these are label maps.
    """
    if moving_img.shape == reference_img.shape and np.allclose(moving_img.affine, reference_img.affine):
        return moving_img
    resampled = niproc.resample_from_to(moving_img, (reference_img.shape, reference_img.affine), order=0)
    return resampled


def component_stats_for_label(
    label_volume: np.ndarray,
    label_value: int,
    totseg_labels: np.ndarray,
    structure: np.ndarray,
    file_id: str,
) -> Tuple[List[Dict], List[int]]:
    """
    Compute per-component statistics for a given `label_value` in `label_volume`.
    Returns:
      - records: list of dict (one per component)
      - sizes_vox: list of lesion sizes (voxels) for histogram aggregation
    """
    mask = label_volume == label_value
    records: List[Dict] = []
    sizes_vox: List[int] = []

    if not np.any(mask):
        return records, sizes_vox

    labeled, ncomp = ndimage.label(mask, structure=structure)
    if ncomp == 0:
        return records, sizes_vox

    for comp_id in range(1, ncomp + 1):
        comp = labeled == comp_id
        voxels = int(np.count_nonzero(comp))
        if voxels == 0:
            continue

        # Extract totseg labels under this component to compute overlaps efficiently
        comp_totseg_vals = totseg_labels[comp]

        # Per-label counts for the requested original totseg labels
        per_label_counts: Dict[int, int] = {}
        for lab in TOTSEG_ORIGINAL_LABELS_TO_REPORT:
            per_label_counts[lab] = int(np.count_nonzero(comp_totseg_vals == lab))

        # Aggregates
        overlap_bone = int(sum(per_label_counts[lab] for lab in TOTSEG_BONE_LABELS))
        overlap_other_remapped = per_label_counts.get(4, 0)  # current remapped "other organs"=4
        overlap_other_original_total = int(sum(per_label_counts[lab] for lab in TOTSEG_OTHER_ORG_ORIGINAL_LABELS))

        rec = {
            "file_id": file_id,
            "label_value": label_value,
            "label_name": "TTB" if label_value == 1 else "Physiologic",
            "instance_id": comp_id,
            "voxel_count": voxels,
            "overlap_bone_voxels": overlap_bone,
            "overlap_other_organs_voxels": overlap_other_remapped,
            "overlap_bone_fraction": float(overlap_bone / voxels),
            "overlap_other_organs_fraction": float(overlap_other_remapped / voxels),
            "overlap_other_organs_original_total_voxels": overlap_other_original_total,
            "overlap_other_organs_original_total_fraction": float(overlap_other_original_total / voxels),
        }

        # Add per-original-totseg-label overlap columns
        for lab in TOTSEG_ORIGINAL_LABELS_TO_REPORT:
            rec[f"overlap_totseg_orig_{lab}_voxels"] = per_label_counts[lab]

        records.append(rec)
        sizes_vox.append(voxels)

    return records, sizes_vox


def analyze_pair(ttb_path: Path, totseg_dir: Path, structure: np.ndarray) -> Tuple[List[Dict], List[int], List[int]]:
    """
    Analyze one TTB/phys NIfTI file and its paired totseg file.
    Returns:
      - per-component records (list of dict)
      - sizes_vox_label1 (TTB) for histogram
      - sizes_vox_label2 (Physiologic) for histogram
    """
    # Load images
    ttb_img = load_nifti(ttb_path)
    ttb_data = np.asarray(ttb_img.get_fdata(), dtype=np.int16)

    totseg_path = find_matching_totseg(ttb_path, totseg_dir)
    if totseg_path is None:
        print(f"[WARN] No matching totseg file found for: {ttb_path.name}")
        return [], [], []

    totseg_img = load_nifti(totseg_path)
    totseg_img_on_ttb = ensure_same_grid(totseg_img, ttb_img)
    totseg_data = np.asarray(totseg_img_on_ttb.get_fdata(), dtype=np.int16)

    file_id = ttb_path.name
    recs1, sizes1 = component_stats_for_label(ttb_data, 1, totseg_data, structure, file_id)
    recs2, sizes2 = component_stats_for_label(ttb_data, 2, totseg_data, structure, file_id)

    return (recs1 + recs2), sizes1, sizes2


def plot_and_save_hist(sizes_vox: List[int], title: str, out_path: Path) -> None:
    """
    Plot and save a histogram of lesion sizes in voxels with x-axis 0..500 and 10-voxel bins.
    """
    if len(sizes_vox) == 0:
        print(f"[INFO] No lesion sizes to plot for: {title}")
        return
    arr = np.asarray(sizes_vox, dtype=int)

    # Info about values outside plotting range
    n_below = int(np.sum(arr < HIST_X_MIN))
    n_above = int(np.sum(arr > HIST_X_MAX))
    if n_below or n_above:
        print(f"[INFO] {title}: {n_below} < {HIST_X_MIN} vox and {n_above} > {HIST_X_MAX} vox will be outside the plotted range.")

    plt.figure()
    plt.hist(arr, bins=HIST_BINS)
    plt.xlabel("Lesion size (voxels)")
    plt.ylabel("Count")
    plt.title(title)
    plt.xlim(HIST_X_MIN, HIST_X_MAX)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # Validate input directories
    if not TTB_DIR.exists():
        raise FileNotFoundError(f"TTB/Phys directory not found: {TTB_DIR}")
    if not TOTSEG_DIR.exists():
        raise FileNotFoundError(f"Totalseg directory not found: {TOTSEG_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(TTB_DIR.glob(FILE_GLOB))
    if len(files) == 0:
        raise FileNotFoundError(f"No NIfTI files found in {TTB_DIR} with pattern {FILE_GLOB}")

    structure = generate_18_structure()

    all_records: List[Dict] = []
    all_sizes_label1: List[int] = []
    all_sizes_label2: List[int] = []

    print(f"[INFO] Found {len(files)} TTB/phys masks. Beginning analysis...")
    for i, fpath in enumerate(files, 1):
        try:
            recs, s1, s2 = analyze_pair(fpath, TOTSEG_DIR, structure)
            all_records.extend(recs)
            all_sizes_label1.extend(s1)
            all_sizes_label2.extend(s2)
            if i % 10 == 0 or i == len(files):
                print(f"[INFO] Processed {i}/{len(files)} files.")
        except Exception as e:
            print(f"[ERROR] Failed on {fpath.name}: {e}")

    # Save per-lesion table
    df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "lesion_instance_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote per-lesion stats: {csv_path}")

    # Histograms (fixed axis and 10-voxel bins)
    plot_and_save_hist(all_sizes_label1, "TTB lesions (label=1): instance sizes (voxels)", OUTPUT_DIR / "hist_ttb_voxels.png")
    plot_and_save_hist(all_sizes_label2, "Physiologic lesions (label=2): instance sizes (voxels)", OUTPUT_DIR / "hist_physiologic_voxels.png")
    print(f"[INFO] Saved histograms in: {OUTPUT_DIR}")

    # Summary stats in voxels
    if len(all_sizes_label1) > 0 or len(all_sizes_label2) > 0:
        stats = []
        if len(all_sizes_label1) > 0:
            v = np.array(all_sizes_label1, dtype=int)
            stats.append({"label_value": 1, "label_name": "TTB", "n_instances": int(v.size),
                          "median_voxels": float(np.median(v)),
                          "mean_voxels": float(np.mean(v)),
                          "p10_voxels": float(np.percentile(v, 10)),
                          "p90_voxels": float(np.percentile(v, 90))})
        if len(all_sizes_label2) > 0:
            v = np.array(all_sizes_label2, dtype=int)
            stats.append({"label_value": 2, "label_name": "Physiologic", "n_instances": int(v.size),
                          "median_voxels": float(np.median(v)),
                          "mean_voxels": float(np.mean(v)),
                          "p10_voxels": float(np.percentile(v, 10)),
                          "p90_voxels": float(np.percentile(v, 90))})
        pd.DataFrame(stats).to_csv(OUTPUT_DIR / "voxel_summary_stats.csv", index=False)
        print(f"[INFO] Wrote summary stats: {OUTPUT_DIR / 'voxel_summary_stats.csv'}")

    print(f"[DONE] Outputs are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
