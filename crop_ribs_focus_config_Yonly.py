#!/usr/bin/env python3
"""
Crop registered PET/CT/TTB/TotalSeg NIfTI volumes to a body-focused FOV
defined by any TotalSeg labels in {1,2,3,4} (assumes TotalSeg already remapped to 0..4).

Changes vs prior version:
- Uses all labels 1..4 (not just ribs=2) to define the ROI.
- Expands ROI in X, Y, and Z (symmetric padding).
- ROI is clamped to original image bounds (never exceeds).
- Keeps affine consistent so world coordinates are preserved.

Requirements
------------
pip install nibabel numpy tqdm
"""

# ============================== USER CONFIG (EDIT HERE) ==============================
from pathlib import Path

# Root PSMA directory and modality subfolders
ROOT = Path("/media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3/FDG")
TOTSEG_DIR = ROOT / "totseg_resampled"
CT_DIR     = ROOT / "CT_resampled"
PET_DIR    = ROOT / "PET_rescaled"
TTB_DIR    = ROOT / "TTB"
TTB_PHYS_DIR = ROOT / "TTB_phys"

# Output naming
OUTPUT_SUFFIX   = "_bodyCrop"  # new sibling folders: <modality><OUTPUT_SUFFIX>/
FILENAME_GLOB   = "train_*.nii.gz"
CSV_LOG_PATH    = ROOT / f"bodyCrop_log{OUTPUT_SUFFIX}.csv"

# Symmetric padding (voxels) applied in ALL dimensions
PAD_X = 10
PAD_Y = 10
PAD_Z = 10

# Label values for 'body' in already-remapped TotalSeg
BODY_LABELS = (1, 2, 3, 4)

# Fallback if no BODY_LABELS found:
#   - True  -> keep full volume (no cropping)
#   - False -> skip the case and record in CSV
FALLBACK_TO_FULL_VOLUME_IF_NONE_FOUND = True
# ============================ END USER CONFIG (EDIT HERE) ============================

import csv
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


def _safe_name_no_niigz(p: Path) -> str:
    """Return filename without the .nii.gz extension (works also if only .nii)."""
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def _compute_body_bbox_from_totseg(arr_ts: np.ndarray, labels=BODY_LABELS) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Compute [xmin, xmax, ymin, ymax, zmin, zmax] for voxels where arr_ts is in `labels`.
    Returns None if not found.
    """
    mask = np.isin(arr_ts, labels)
    if not np.any(mask):
        return None
    xs, ys, zs = np.where(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(zs.min()), int(zs.max())


def _bbox_to_slices_expand_all(
    bbox: Tuple[int, int, int, int, int, int],
    shape: Tuple[int, int, int],
    pad_x: int,
    pad_y: int,
    pad_z: int,
) -> Tuple[slice, slice, slice]:
    """
    Convert bbox to numpy slices with symmetric padding in all dims and clamp to bounds.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    X, Y, Z = shape

    new_xmin = max(0, x_min - pad_x)
    new_xmax = min(X - 1, x_max + pad_x)

    new_ymin = max(0, y_min - pad_y)
    new_ymax = min(Y - 1, y_max + pad_y)

    new_zmin = max(0, z_min - pad_z)
    new_zmax = min(Z - 1, z_max + pad_z)

    return (
        slice(new_xmin, new_xmax + 1),
        slice(new_ymin, new_ymax + 1),
        slice(new_zmin, new_zmax + 1),
    )


def _crop_img_keep_affine(img: nib.Nifti1Image, slc: Tuple[slice, slice, slice]) -> nib.Nifti1Image:
    """
    Crop NIfTI by voxel slices while preserving spatial coordinates (qform/sform updated).
    """
    data = np.asanyarray(img.dataobj)[slc]
    i0, j0, k0 = slc[0].start, slc[1].start, slc[2].start

    A = img.affine.copy()
    offset = np.array([i0, j0, k0], dtype=float)
    new_affine = A.copy()
    new_affine[:3, 3] = A[:3, :3].dot(offset) + A[:3, 3]

    hdr = img.header.copy()
    out = nib.Nifti1Image(data, new_affine, header=hdr)

    Q, qcode = img.get_qform(coded=True)
    S, scode = img.get_sform(coded=True)
    out.set_qform(new_affine, int(qcode) if qcode is not None else 1)
    out.set_sform(new_affine, int(scode) if scode is not None else 1)
    return out


def _ensure_same_shape(ref_shape: Tuple[int, int, int], shape: Tuple[int, int, int], label: str, case: str) -> None:
    if ref_shape != shape:
        raise ValueError(
            f"[{case}] {label} shape {shape} does not match TotalSeg shape {ref_shape}. "
            "All modalities must be on the same voxel grid."
        )


def _save_csv_log(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Validate input dirs
    for d in [TOTSEG_DIR, CT_DIR, PET_DIR, TTB_DIR, TTB_PHYS_DIR]:
        if not d.exists():
            raise FileNotFoundError(f"Input folder does not exist: {d}")

    out_dirs = {
        "totseg": Path(str(TOTSEG_DIR) + OUTPUT_SUFFIX),
        "ct":     Path(str(CT_DIR)     + OUTPUT_SUFFIX),
        "pet":    Path(str(PET_DIR)    + OUTPUT_SUFFIX),
        "ttb":    Path(str(TTB_DIR)    + OUTPUT_SUFFIX),
        "ttb_phys": Path(str(TTB_PHYS_DIR) + OUTPUT_SUFFIX),
    }
    for od in out_dirs.values():
        od.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    cases = sorted(TOTSEG_DIR.glob(FILENAME_GLOB))
    if not cases:
        raise RuntimeError(f"No cases found in {TOTSEG_DIR} matching '{FILENAME_GLOB}'")

    for ts_path in tqdm(cases, desc="Cropping (tight bounding box over labels 1..4)", unit="case"):
        case_id = _safe_name_no_niigz(ts_path)
        out_rec: Dict[str, object] = {"case": case_id}

        paths = {
            "totseg": ts_path,
            "ct":     CT_DIR / ts_path.name,
            "pet":    PET_DIR / ts_path.name,
            "ttb":    TTB_DIR / ts_path.name,
            "ttb_phys": TTB_PHYS_DIR / ts_path.name,
        }

        missing = [k for k, v in paths.items() if not v.exists()]
        if missing:
            out_rec["status"] = f"SKIP: missing {missing}"
            rows.append(out_rec)
            continue

        try:
            # Load TotalSeg and compute bbox over labels 1..4
            ts_img  = nib.load(str(paths["totseg"]))
            ts_data = np.asanyarray(ts_img.dataobj)
            ref_shape = ts_data.shape
            out_rec.update({"shape_x": ref_shape[0], "shape_y": ref_shape[1], "shape_z": ref_shape[2]})

            bbox = _compute_body_bbox_from_totseg(ts_data, labels=BODY_LABELS)
            if bbox is None:
                if not FALLBACK_TO_FULL_VOLUME_IF_NONE_FOUND:
                    out_rec["status"] = "SKIP: no labels 1..4 found"
                    rows.append(out_rec)
                    continue
                X, Y, Z = ref_shape
                bbox = (0, X - 1, 0, Y - 1, 0, Z - 1)
                out_rec["bbox_source"] = "full_volume (labels 1..4 not found)"
            else:
                out_rec["bbox_source"] = "totseg_labels_1to4"

            slc = _bbox_to_slices_expand_all(
                bbox=bbox,
                shape=ref_shape,
                pad_x=PAD_X,
                pad_y=PAD_Y,
                pad_z=PAD_Z,
            )
            out_rec.update({
                "x_start": slc[0].start, "x_stop": slc[0].stop,
                "y_start": slc[1].start, "y_stop": slc[1].stop,
                "z_start": slc[2].start, "z_stop": slc[2].stop,
                "pad_x": PAD_X, "pad_y": PAD_Y, "pad_z": PAD_Z,
            })

            # Ensure shapes match across modalities
            ct_img  = nib.load(str(paths["ct"]));  _ensure_same_shape(ref_shape, ct_img.shape,  "CT",  case_id)
            pet_img = nib.load(str(paths["pet"])); _ensure_same_shape(ref_shape, pet_img.shape, "PET", case_id)
            ttb_img = nib.load(str(paths["ttb"])); _ensure_same_shape(ref_shape, ttb_img.shape, "TTB", case_id)
            ttb_phys_img = nib.load(str(paths["ttb_phys"])); _ensure_same_shape(ref_shape, ttb_phys_img.shape, "TTB_PHYS", case_id)

            # Crop & save
            nib.save(_crop_img_keep_affine(ts_img,  slc), str(out_dirs["totseg"] / ts_path.name))
            nib.save(_crop_img_keep_affine(ct_img,  slc), str(out_dirs["ct"]     / ts_path.name))
            nib.save(_crop_img_keep_affine(pet_img, slc), str(out_dirs["pet"]    / ts_path.name))
            nib.save(_crop_img_keep_affine(ttb_img, slc), str(out_dirs["ttb"]    / ts_path.name))
            nib.save(_crop_img_keep_affine(ttb_phys_img, slc), str(out_dirs["ttb_phys"] / ts_path.name))

            out_rec["status"] = "OK"

        except Exception as e:
            out_rec["status"] = f"ERROR: {type(e).__name__}: {e}"

        rows.append(out_rec)

    _save_csv_log(rows, CSV_LOG_PATH)
    print(f"Done. CSV log written to: {CSV_LOG_PATH}")
    print("Output folders:")
    for k, v in out_dirs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
