import shutil
import csv
import json
from pathlib import Path

# ==========================
# CONFIG
# ==========================

ANNOTATION_DIR = Path("/Volumes/T7 Shield/json_stuff/renamed_json")
DATASET_ROOT = Path("/Volumes/T7 Shield/VISTA_dataset_blurred")

LOG_DIR = DATASET_ROOT / "annotation_logs"
LOG_DIR.mkdir(exist_ok=True)

TXT_LOG = LOG_DIR / "annotation_log.txt"
CSV_LOG = LOG_DIR / "annotation_status.csv"

# ==========================
# TRACKING
# ==========================

records = []

inserted = 0
missing_clip = 0
missing_json = 0

# ==========================
# INSERT ANNOTATIONS
# ==========================

for ann_file in sorted(ANNOTATION_DIR.glob("*.json")):

    clip_id = ann_file.stem
    tc_id = clip_id.split("_")[0]

    clip_folder = DATASET_ROOT / tc_id / clip_id
    target_json = clip_folder / f"{clip_id}.json"

    if not clip_folder.exists():
        records.append([clip_id, "missing_clip_folder"])
        missing_clip += 1
        continue

    if not target_json.exists():
        records.append([clip_id, "missing_json"])
        missing_json += 1
        continue

    try:

        shutil.copy2(ann_file, target_json)

        inserted += 1
        records.append([clip_id, "annotation_inserted"])

    except Exception:
        records.append([clip_id, "error"])

# ==========================
# DATASET STATISTICS
# ==========================

total_clips = 0
annotated = 0
empty_json = 0

for tc_folder in sorted(DATASET_ROOT.glob("TC*")):

    if not tc_folder.is_dir():
        continue

    for clip_folder in tc_folder.glob("TC*"):

        if not clip_folder.is_dir():
            continue

        total_clips += 1

        clip_id = clip_folder.name
        json_file = clip_folder / f"{clip_id}.json"

        if not json_file.exists():
            empty_json += 1
            continue

        try:

            if json_file.stat().st_size < 5:
                empty_json += 1
                continue

            with open(json_file) as f:
                data = json.load(f)

            if data == {} or data == []:
                empty_json += 1
            else:
                annotated += 1

        except Exception:
            empty_json += 1

coverage = 0
if total_clips > 0:
    coverage = (annotated / total_clips) * 100

# ==========================
# WRITE CSV LOG
# ==========================

with open(CSV_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_id", "status"])
    writer.writerows(records)

# ==========================
# WRITE TXT LOG
# ==========================

with open(TXT_LOG, "w") as f:

    f.write("VISTA Annotation Insert Log\n")
    f.write("===========================\n\n")

    for clip_id, status in records:
        f.write(f"{clip_id} : {status}\n")

    f.write("\nDataset Statistics\n")
    f.write("------------------\n")

    f.write(f"Total clips: {total_clips}\n")
    f.write(f"Annotated clips: {annotated}\n")
    f.write(f"Empty JSON files: {empty_json}\n")
    f.write(f"Coverage: {coverage:.1f}%\n")

print("Logs saved:")
print(TXT_LOG)
print(CSV_LOG)