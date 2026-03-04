#!/usr/bin/env python3
import csv
import hashlib
import random
import shutil
import re
from pathlib import Path
from collections import defaultdict

# ==================================================
# CONFIG
# ==================================================
INPUT_DIR = Path("/Volumes/T7 Shield/VISTA_dataset_blurred")
OUTPUT_DIR = Path("/Volumes/T7 Shield/annotation_batches")

ANNOTATORS = 8
ROUNDS = 2
OVERLAP_PER_TC_PER_ROUND = 3   # shared across all annotators
SEED = 42
DRY_RUN = False

TC_REGEX = r"(TC\d{2})"

# ==================================================
# Helpers
# ==================================================
def extract_tc(path: Path):
    for part in path.parts:
        if part.startswith("TC") and part[2:].isdigit():
            return part
    return "UNKNOWN"


def file_hash(path, algo="sha1", chunk=8192):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()

# ==================================================
# Main
# ==================================================
def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_DIR.rglob("*_blurred.mp4"))
    if not files:
        raise RuntimeError("No MP4 files found")

    print(f"🎥 Total videos: {len(files)}")

    # ----------------------------------------------
    # Group by test case
    # ----------------------------------------------
    by_tc = defaultdict(list)
    for f in files:
        by_tc[extract_tc(f)].append(f)

    tcs = sorted(by_tc.keys())
    print(f"🧪 Test cases: {tcs}")

    # ----------------------------------------------
    # Split unique files across rounds
    # ----------------------------------------------
    for tc in tcs:
        random.shuffle(by_tc[tc])

    rounds = [defaultdict(list) for _ in range(ROUNDS)]

    for tc, tc_files in by_tc.items():
        print(f"{tc}: {len(tc_files)} blurred videos")
        chunk_size = len(tc_files) // ROUNDS
        for r in range(ROUNDS):
            start = r * chunk_size
            end = len(tc_files) if r == ROUNDS - 1 else (r + 1) * chunk_size
            rounds[r][tc].extend(tc_files[start:end])

    annotators = [f"annotator_{i+1}" for i in range(ANNOTATORS)]
    log_rows = []

    # ----------------------------------------------
    # Build each round
    # ----------------------------------------------
    for r, tc_map in enumerate(rounds, start=1):
        print(f"\n🔵 Building round {r}")
        round_dir = OUTPUT_DIR / f"round_{r}"
        round_dir.mkdir(exist_ok=True)

        assignments = {a: [] for a in annotators}

        for tc, tc_files in tc_map.items():
            random.shuffle(tc_files)

            if len(tc_files) < OVERLAP_PER_TC_PER_ROUND:
                raise RuntimeError(f"{tc}: not enough files for overlap")

            # ---- overlap ----
            offset = (r - 1) * OVERLAP_PER_TC_PER_ROUND
            overlap = tc_files[offset : offset + OVERLAP_PER_TC_PER_ROUND]
            remaining = [
                f for i, f in enumerate(tc_files)
                if i < offset or i >= offset + OVERLAP_PER_TC_PER_ROUND
            ]


            for a in annotators:
                assignments[a].extend(overlap)

            # ---- distribute remaining round-robin ----
            for i, f in enumerate(remaining):
                assignments[annotators[i % ANNOTATORS]].append(f)

        # ---- copy files + log ----
        for a, vids in assignments.items():
            dest = round_dir / a
            dest.mkdir(exist_ok=True)

            for v in vids:
                if not DRY_RUN:
                    shutil.copy2(v, dest / v.name)

                log_rows.append([
                    r,
                    a,
                    v.name,
                    extract_tc(v),
                    file_hash(v)
                ])

            print(f"  👤 {a}: {len(vids)} videos")

    # ----------------------------------------------
    # Write CSV log
    # ----------------------------------------------
    csv_path = OUTPUT_DIR / "assignment_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "annotator",
            "filename",
            "test_case",
            "sha1"
        ])
        writer.writerows(log_rows)

    print("\nAssignment complete")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Log: {csv_path}")

if __name__ == "__main__":
    main()
