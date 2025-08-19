#!/usr/bin/env python3
"""
PSMA + FDG PET-only inference with STAPLE ensemble.

Outputs per tracer (PSMA or FDG):
  - *_ttb_phys.mha    (all voxels ≥ SUVthr)
  - *_physio.mha      (STAPLE ensemble of folds, label=2)
  - *_physio_exp.mha  (physio mask expanded by 1 voxel)
  - *_ttb.mha         (TTB_phys minus physio_exp mask)

Also:
  - Computes Dice for physio (vs GT label=2) and TTB (vs GT label=1).
  - Prints per-case and summary Dice scores to console.
  - Writes summary CSV with all Dice scores and means per tracer.
"""

import os, sys, json, shutil, warnings, csv
import SimpleITK as sitk
import numpy as np
from typing import List

# ────────────── Paths ──────────────
BASE     = "/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/DeepPSMAAutoSeg"
CHALLENGE_DATA = "/media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/CHALLENGE_DATA"
GT_BASE  = "/media/dlabella29/Extreme Pro/Grand Challenge Data/DEEP-PSMA/DEEP-PSMA_CHALLENGE_DATA/data3"

BUNDLES = {
    "PSMA": os.path.join(BASE, "PSMA_workdir_phys_PETonly"),  # segresnet_0 … 4
    "FDG":  os.path.join(BASE, "FDG_workdir_phys_PETonly"),   # segresnet_0 … 4
}
TMP      = os.path.join(BASE, "tmp_physio")
OUTPUT   = os.path.join(BASE, "output", "images")
NUM_FOLDS = 4

# ────────────── Utilities ──────────────
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def clean_dir(p: str): shutil.rmtree(p, ignore_errors=True); os.makedirs(p, exist_ok=True)

def rescale_pet(pet_fp: str, thr_fp: str, out_fp: str):
    pet_img = sitk.ReadImage(pet_fp)
    with open(thr_fp, "r") as f:
        suv_thr = float(json.load(f)["suv_threshold"])
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

def subtract_physio_from_ttb(ttb_phys: sitk.Image, physio_mask: sitk.Image) -> (sitk.Image, sitk.Image):
    physio_exp = expand_mask(physio_mask, radius=1)
    ttb_arr = sitk.GetArrayFromImage(ttb_phys) > 0
    physio_arr = sitk.GetArrayFromImage(physio_exp) > 0
    final_arr = np.logical_and(ttb_arr, np.logical_not(physio_arr)).astype(np.uint8)
    out = sitk.GetImageFromArray(final_arr)
    out.CopyInformation(ttb_phys)
    return out, physio_exp

def dice_score(pred: sitk.Image, gt: sitk.Image, label_val: int) -> float:
    pred_arr = (sitk.GetArrayFromImage(pred) == label_val).astype(np.uint8)
    gt_arr   = (sitk.GetArrayFromImage(gt) == label_val).astype(np.uint8)
    inter = np.sum(pred_arr * gt_arr)
    denom = np.sum(pred_arr) + np.sum(gt_arr)
    if denom == 0: return 1.0
    return 2.0 * inter / denom

# ────────────── Pipeline ──────────────
def process_case(tracer: str, case_dir: str, results: list):
    out_dir = os.path.join(OUTPUT, tracer.lower() + "-final")
    ensure_dir(out_dir)
    ensure_dir(TMP)
    uuid = os.path.basename(os.path.dirname(case_dir))  # e.g. train_0001
    print(f"\n▶ Processing {tracer} case: {uuid}")

    pet_fp = os.path.join(case_dir, "PET.nii.gz")
    thr_fp = os.path.join(case_dir, "threshold.json")
    case_tmp = os.path.join(TMP, f"{uuid}_{tracer}")
    clean_dir(case_tmp)

    rescaled_pet_fp = os.path.join(case_tmp, f"{uuid}_0000.mha")
    pet_rescaled, suv_thr = rescale_pet(pet_fp, thr_fp, rescaled_pet_fp)

    ttb_phys_fp = os.path.join(case_tmp, f"{uuid}_ttb_phys.mha")
    ttb_phys = create_ttb_phys(pet_rescaled, ttb_phys_fp)

    datalist_json = os.path.join(case_tmp, f"{uuid}_test.json")
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

    physio_fp = os.path.join(out_dir, f"{uuid}_physio.mha")
    sitk.WriteImage(physio_mask, physio_fp, useCompression=True)

    ttb_final, physio_exp = subtract_physio_from_ttb(ttb_phys, physio_mask)

    physio_exp_fp = os.path.join(out_dir, f"{uuid}_physio_exp.mha")
    sitk.WriteImage(physio_exp, physio_exp_fp, useCompression=True)

    ttb_fp = os.path.join(out_dir, f"{uuid}_ttb.mha")
    sitk.WriteImage(ttb_final, ttb_fp, useCompression=True)

    # Dice vs GT
    gt_fp = os.path.join(GT_BASE, tracer, f"{uuid}.nii.gz")
    if os.path.exists(gt_fp):
        gt_img = sitk.ReadImage(gt_fp)
        dice_phys = dice_score(physio_mask, gt_img, label_val=2)
        dice_ttb  = dice_score(ttb_final,  gt_img, label_val=1)
        print(f"  Dice {tracer} Physio vs GT: {dice_phys:.4f}")
        print(f"  Dice {tracer} TTB    vs GT: {dice_ttb:.4f}")
        results.append((uuid, dice_phys, dice_ttb))
    else:
        print(f"⚠️ No GT file found for {uuid} ({tracer})")
        results.append((uuid, None, None))

# ────────────── Entry point ──────────────
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    for tracer in ["PSMA", "FDG"]:
        print(f"\n===== Running Inference for {tracer} =====")
        results = []
        for case in sorted(os.listdir(CHALLENGE_DATA)):
            tracer_dir = os.path.join(CHALLENGE_DATA, case, tracer)
            if os.path.isdir(tracer_dir) and os.path.exists(os.path.join(tracer_dir, "PET.nii.gz")):
                process_case(tracer, tracer_dir, results)

        # Console summary
        print(f"\n==== {tracer} Dice Summary ====")
        phys_vals = [r[1] for r in results if r[1] is not None]
        ttb_vals  = [r[2] for r in results if r[2] is not None]
        for uuid, dp, dt in results:
            print(f"{uuid}: Physio={dp if dp is not None else 'NA'} , TTB={dt if dt is not None else 'NA'}")
        if phys_vals and ttb_vals:
            print(f"\nMean Dice {tracer} Physio: {np.mean(phys_vals):.4f}")
            print(f"Mean Dice {tracer} TTB:    {np.mean(ttb_vals):.4f}")

        # CSV summary
        out_dir = os.path.join(OUTPUT, tracer.lower() + "-final")
        csv_fp = os.path.join(out_dir, f"{tracer.lower()}_ttb_physio_dice_summary.csv")
        with open(csv_fp, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Case", "Dice_Physio", "Dice_TTB"])
            for uuid, dp, dt in results:
                writer.writerow([uuid, dp if dp is not None else "NA", dt if dt is not None else "NA"])
            if phys_vals and ttb_vals:
                writer.writerow([])
                writer.writerow(["Mean", np.mean(phys_vals), np.mean(ttb_vals)])
        print(f"\n✅ {tracer} Dice summary saved to {csv_fp}")

if __name__ == "__main__":
    main()
