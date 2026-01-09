import csv
import shutil
from pathlib import Path

# CHANGE THESE PATHS
DATASET_DIR = Path("dataset_raw")
OUTPUT_DIR = Path("dataset_renamed")
CSV_PATH = Path("rename_map.csv")

OUTPUT_DIR.mkdir(exist_ok=True)

with open(CSV_PATH, newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        original_file = DATASET_DIR / row["original_filename"]
        test_case = row["test_case"]
        index = int(row["index"])

        tc_dir = OUTPUT_DIR / test_case
        tc_dir.mkdir(exist_ok=True)

        new_name = f"{test_case}_{index:03d}{original_file.suffix}"
        new_path = tc_dir / new_name

        if not original_file.exists():
            print(f"❌ Missing: {original_file}")
            continue

        shutil.copy2(original_file, new_path)
        print(f"✅ {original_file.name} → {new_path}")
