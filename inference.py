#!/usr/bin/env python3
"""
PSMA + FDG inference with STAPLE ensemble.

Input folder structure:
input/
├── fdg-pet-suv-threshold.json   (or float file)
├── psma-pet-suv-threshold.json  (or float file)
└── images/
    ├── fdg-pet/*.mha
    └── psma-pet-ga-68/*.mha

Output (final only):
  - /output/images/psma-pet-ttb/<basename>_ttb.mha
  - /output/images/fdg-pet-ttb/<basename>_ttb.mha

All intermediate files go to /tmp.
"""

import os, sys, json, shutil, warnings, glob
import SimpleITK as sitk
import numpy as np
from typing import List

# ────────────── Paths ──────────────
BASE        = "/"
INPUT_ROOT  = os.path.join(BASE, "input")
TMP         = os.path.join(BASE, "tmp")
OUTPUT      = os.path.join(BASE, "output", "images")
NUM_FOLDS   = 5

BUNDLES = {
    "PSMA": os.path.join(BASE, "PSMA_workdir_phys_PETonly"),
    "FDG":  os.path.join(BASE, "FDG_workdir_phys_PETonly"),
}
OUT_DIRS = {
    "PSMA": os.path.join(OUTPUT, "psma-pet-ttb"),
    "FDG":  os.path.join(OUTPUT, "fdg-pet-ttb"),
}
PET_DIRS = {
    "PSMA": os.path.join(INPUT_ROOT, "images", "psma-pet-ga-68"),
    "FDG":  os.path.join(INPUT_ROOT, "images", "fdg-pet"),
}
THR_FILES = {
    "PSMA": os.path.join(INPUT_ROOT, "psma-pet-suv-threshold.json"),
    "FDG":  os.path.join(INPUT_ROOT, "fdg-pet-suv-threshold.json"),
}

# ────────────── Utilities ──────────────
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def clean_dir(p: str): shutil.rmtree(p, ignore_errors=True); os.makedirs(p, exist_ok=True)

def load_threshold(thr_fp: str) -> float:
    """Load SUV threshold from JSON with 'suv_threshold' or plain float file."""
    with open(thr_fp, "r") as f:
        content = f.read().strip()
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "suv_threshold" in obj:
                return float(obj["suv_threshold"])
            else:
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            return float(content)

def rescale_pet(pet_fp: str, thr_fp: str, out_fp: str):
    pet_img = sitk.ReadImage(pet_fp)
    suv_thr = load_threshold(thr_fp)
    pet_rescaled = sitk.Cast(pet_img, sitk.sitkFloat32) / suv_thr
    pet_rescaled.CopyInformation(pet_img)
    sitk.WriteImage(pet_rescaled, out_fp, useCompression=True)
    return pet_rescaled, suv_thr

def create_ttb_phys(pet_rescaled: sitk.Image, out_fp: str):
    arr = sitk.GetArrayFromImage(pet_rescaled)
    mask_arr = (arr >= 1.0).astype("uint8")
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(pet_rescaled)
    sitk.WriteImage(mask_img, out_fp, useCompression=True)
    return mask_img

def run_bundle_infer(config_file: str, bundle_root: str,
                     data_base_dir: str, datalist_json: str) -> str:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from segmenter import run_segmenter
    pred_dir = os.path.join(bundle_root, "prediction_testing")
    if os.path.isdir(pred_dir): shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    override = {
        "bundle_root": bundle_root,
        "data_file_base_dir": data_base_dir,
        "data_list_file_path": datalist_json,
        "infer#enabled": True,
    }
    run_segmenter(config_file=config_file, **override)
    preds = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
    if not preds: raise RuntimeError(f"No predictions found in {pred_dir}")
    return preds[0]

def expand_mask(mask: sitk.Image, radius: int = 1) -> sitk.Image:
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(radius)
    dilate.SetForegroundValue(1)
    return dilate.Execute(mask)

def subtract_physio_from_ttb(ttb_phys: sitk.Image, physio_mask: sitk.Image):
    physio_exp = expand_mask(physio_mask, radius=1)
    ttb_arr = sitk.GetArrayFromImage(ttb_phys) > 0
    physio_arr = sitk.GetArrayFromImage(physio_exp) > 0
    final_arr = np.logical_and(ttb_arr, np.logical_not(physio_arr)).astype(np.uint8)
    out = sitk.GetImageFromArray(final_arr)
    out.CopyInformation(ttb_phys)
    return out

# ────────────── Pipeline ──────────────
def process_case(tracer: str, pet_fp: str, thr_fp: str):
    out_dir = OUT_DIRS[tracer]
    ensure_dir(out_dir)
    ensure_dir(TMP)

    case_id = os.path.splitext(os.path.basename(pet_fp))[0]
    case_tmp = os.path.join(TMP, f"{case_id}_{tracer}")
    clean_dir(case_tmp)

    rescaled_pet_fp = os.path.join(case_tmp, f"{case_id}_0000.mha")
    pet_rescaled, suv_thr = rescale_pet(pet_fp, thr_fp, rescaled_pet_fp)
    print(f"   {tracer} [{case_id}] SUV threshold = {suv_thr:.4f}")

    ttb_phys_fp = os.path.join(case_tmp, f"{case_id}_ttb_phys.mha")
    ttb_phys = create_ttb_phys(pet_rescaled, ttb_phys_fp)

    datalist_json = os.path.join(case_tmp, f"{case_id}_test.json")
    with open(datalist_json, "w") as f:
        json.dump({"testing": [{"image": [os.path.basename(rescaled_pet_fp)]}]}, f, indent=2)

    pred_paths: List[str] = []
    for fold in range(NUM_FOLDS):
        cfg = os.path.join(BUNDLES[tracer], f"segresnet_{fold}", "configs", "hyper_parameters.yaml")
        bundle_root = os.path.dirname(os.path.dirname(cfg))
        pred_src = run_bundle_infer(cfg, bundle_root, case_tmp, datalist_json)
        fold_dst = os.path.join(case_tmp, f"pred_fold{fold}.nii.gz")
        shutil.copy2(pred_src, fold_dst)
        pred_paths.append(fold_dst)

    # STAPLE fusion
    binary_preds = []
    for p in pred_paths:
        seg = sitk.ReadImage(p)
        arr = sitk.GetArrayFromImage(seg)
        mask_arr = (arr == 2).astype("uint8")
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(seg)
        binary_preds.append(mask_img)

    staple_filter = sitk.STAPLEImageFilter()
    staple_prob = staple_filter.Execute(binary_preds)
    physio_mask = sitk.Cast(staple_prob >= 0.5, sitk.sitkUInt8)

    # Final TTB = thresholded PET minus physio
    ttb_final = subtract_physio_from_ttb(ttb_phys, physio_mask)

    # Save ONLY the final TTB to output
    ttb_fp = os.path.join(out_dir, f"{case_id}_ttb.mha")
    sitk.WriteImage(ttb_final, ttb_fp, useCompression=True)

# ────────────── Entry point ──────────────
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    for tracer in ["PSMA", "FDG"]:
        pet_dir = PET_DIRS[tracer]
        thr_fp  = THR_FILES[tracer]
        files   = sorted(glob.glob(os.path.join(pet_dir, "*.mha")))

        if not files:
            print(f"⚠ No PET files found for {tracer} in {pet_dir}")
            continue
        if not os.path.exists(thr_fp):
            print(f"⚠ Threshold file missing for {tracer}: {thr_fp}")
            continue

        print(f"\n▶ Processing {tracer} ({len(files)} PET cases)")
        for pet_fp in files:
            process_case(tracer, pet_fp, thr_fp)

    print("\n✅ Inference complete. Final outputs in /output/images/psma-pet-ttb and /output/images/fdg-pet-ttb")

if __name__ == "__main__":
    main()

