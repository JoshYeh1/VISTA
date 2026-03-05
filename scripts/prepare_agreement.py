import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import string

EXPORT_DIR = Path("/Volumes/T7 Shield/json_files")
CSV_FILE = Path("/Volumes/T7 Shield/annotation_batches/assignment_log.csv")
OUTPUT_DIR = Path("/Volumes/T7 Shield/json_stuff/annotator_agreement")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading assignment log...")

df = pd.read_csv(CSV_FILE)

# --------------------------------
# FIX 1: restrict to round 1
# --------------------------------
if "round" in df.columns:
    df = df[df["round"] == 1]

# remove _blurred.mp4 to get clip id
df["clip_id"] = df["filename"].str.replace("_blurred.mp4", "", regex=False)

# find overlapping clips
overlap_ids = df["clip_id"].value_counts()
overlap_ids = overlap_ids[overlap_ids > 1].index.tolist()

print(f"Found {len(overlap_ids)} overlapping clips")

print("\nScanning annotator exports...\n")

clip_annotations = defaultdict(list)

for json_file in EXPORT_DIR.glob("*.json"):

    # Skip macOS metadata files
    if json_file.name.startswith("._"):
        continue

    print(f"Reading {json_file.name}")

    with open(json_file) as f:
        data = json.load(f)

    for item in data:

        # --------------------------------
        # FIX 2: extract clip_id from Label Studio path
        # --------------------------------
        video_path = item.get("data", {}).get("video", "")
        filename = Path(video_path).name

        # remove random Label Studio upload prefix
        filename = filename.split("-", 1)[-1]

        if not filename.endswith("_blurred.mp4"):
            continue

        clip_id = filename.replace("_blurred.mp4", "")

        if clip_id in overlap_ids:
            clip_annotations[clip_id].append(item)

print("\nWriting condensed annotations...\n")

letters = list(string.ascii_uppercase)

clips_written = 0
annotations_written = 0

for clip_id, anns in clip_annotations.items():

    for i, ann in enumerate(anns):

        letter = letters[i]

        out_file = OUTPUT_DIR / f"{clip_id}_{letter}.json"

        with open(out_file, "w") as f:
            json.dump(ann, f, indent=2)

        annotations_written += 1

    clips_written += 1

print("\nDone.")
print(f"Clips processed: {clips_written}")
print(f"Annotations written: {annotations_written}")
print(f"Output directory: {OUTPUT_DIR}")