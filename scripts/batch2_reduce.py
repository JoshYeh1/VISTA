import csv
import random
import shutil
from pathlib import Path

LOG_PATH = Path("/Volumes/T7 Shield/annotation_batches/assignment_log.csv")
INPUT_DIR = Path("/Volumes/T7 Shield/VISTA_dataset_blurred")
OUTPUT_DIR = Path("/Volumes/T7 Shield/annotation_batches/batch_2_reduced")

ANNOTATORS = 8
SEED = 42

random.seed(SEED)

# ----------------------------------------
# 1. Extract UNIQUE batch 2 filenames
# ----------------------------------------
unique_files = set()

with open(LOG_PATH, newline="") as f:
    reader = csv.DictReader(f)
    print("CSV columns:", reader.fieldnames)  # sanity check
    
    for row in reader:
        if row["round"] == "2":
            unique_files.add(row["filename"])

unique_files = list(unique_files)
print("Unique Batch 2 files:", len(unique_files))

# ----------------------------------------
# 2. Redistribute evenly
# ----------------------------------------
random.shuffle(unique_files)

annotators = [f"annotator_{i+1}" for i in range(ANNOTATORS)]
assignments = {a: [] for a in annotators}

for i, fname in enumerate(unique_files):
    assignments[annotators[i % ANNOTATORS]].append(fname)

# ----------------------------------------
# 3. Rebuild folders (clean first)
# ----------------------------------------
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for a, files in assignments.items():
    dest = OUTPUT_DIR / a
    dest.mkdir(exist_ok=True)

    for fname in files:
        matches = list(INPUT_DIR.rglob(fname))
        if not matches:
            print(f"WARNING: {fname} not found")
            continue

        source = matches[0]
        shutil.copy2(source, dest / fname)

    print(a, "→", len(files))