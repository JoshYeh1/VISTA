#!/usr/bin/env python3
"""
color_correct.py — Simple color‑correction utilities for Meta Aria (or any RGB) images

Two modes:
1. Histogram matching to a "gold‑standard" reference frame (best accuracy).
2. Fully automatic CLAHE + Grey‑World balancing (no reference needed).

Usage examples
--------------

# 1) Histogram‑match every image in a folder to a single reference
python color_correct.py \
    --mode histogram \
    --reference images/ref_office.jpg \
    --input_dir images/raw_office \
    --output_dir images/corrected_office

# 2) Automatic correction when lighting varies
python color_correct.py \
    --mode auto \
    --input_dir images/raw_var \
    --output_dir images/corrected_var

Dependencies
------------
- Python ≥ 3.8
- opencv‑python
- numpy
- scikit‑image (for match_histograms)

Install via:
    pip install opencv-python numpy scikit-image
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage.exposure import match_histograms

############################################################
# Core algorithms
############################################################

def histogram_match(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return *img* with its RGB histogram matched to *ref*."""
    return match_histograms(img, ref, channel_axis=-1)


def clahe_greyworld(img: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast‑Limited AHE on L* + Grey‑World white balance."""
    # --- CLAHE on L* channel (Lab colourspace) ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Grey‑World WB ---
    avg_rgb = img_clahe.mean(axis=(0, 1))
    gain = avg_rgb.mean() / avg_rgb
    balanced = np.clip(img_clahe * gain, 0, 255).astype(np.uint8)
    return balanced

############################################################
# Batch helpers
############################################################

def process_image(path_in: Path, path_out: Path, func):
    img = cv2.imread(str(path_in))
    if img is None:
        print(f"[WARN] Could not read {path_in}")
        return
    corrected = func(img)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path_out), corrected)
    print(f"[OK] {path_out}")


def file_list(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    for p in directory.rglob("*"):
        if p.suffix.lower() in exts:
            yield p

############################################################
# CLI
############################################################

def parse_args():
    ap = argparse.ArgumentParser(description="Fast color correction for image datasets")
    ap.add_argument("--mode", choices=["histogram", "auto"], required=True,
                    help="Correction mode: histogram (needs --reference) or auto (CLAHE+GreyWorld)")
    ap.add_argument("--reference", type=str, default=None, help="Reference image for histogram mode")
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with images to correct")
    ap.add_argument("--output_dir", type=str, required=True, help="Destination folder for corrected images")
    ap.add_argument("--single", type=str, help="Optional single file to correct instead of a directory")
    return ap.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "histogram":
        if args.reference is None:
            sys.exit("--reference is required for histogram mode")
        ref_img = cv2.imread(str(Path(args.reference).expanduser()))
        if ref_img is None:
            sys.exit(f"Could not read reference image: {args.reference}")
        func = lambda img: histogram_match(img, ref_img)
    else:  # auto mode
        func = clahe_greyworld

    if args.single:
        inp = Path(args.single).expanduser()
        outp = output_dir / inp.name
        process_image(inp, outp, func)
    else:
        for p in file_list(input_dir):
            out_path = output_dir / p.relative_to(input_dir)
            process_image(p, out_path, func)


if __name__ == "__main__":
    main()
