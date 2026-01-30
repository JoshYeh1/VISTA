#OG file renaming code. Might be good to keep in case we want to get annotation started 

import pandas as pd
import shutil
from pathlib import Path

CSV_PATH = Path("/Users/joshuayeh/Documents/VISTA Dataset Listings - Collected Data.csv")
RAW_DIR = Path("/Volumes/T7 Shield/all_mp4_video")
OUT_DIR = Path("/Volumes/T7 Shield/VISTA_clean")
LOG_PATH = Path("/Volumes/T7 Shield/rename_log.csv")

OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["File Name", "Case ID No"])
df = df.sort_values(by=["Case ID No", "File Name"])

log_rows = []

for case_id, group in df.groupby("Case ID No"):
    tc_name = f"TC{int(float(case_id)):02d}"
    tc_dir = OUT_DIR / tc_name
    tc_dir.mkdir(exist_ok=True)

    idx = 1
    for _, row in group.iterrows():
        original_name = Path(row["File Name"]).stem + ".mp4"
        original = RAW_DIR / original_name

        if not original.exists():
            print(f"⚠️ Missing MP4: {original_name}")
            log_rows.append(
                {
                    "old_filename": original_name,
                    "new_filename": "",
                    "test_case": tc_name,
                    "status": "missing",
                }
            )
            continue

        new_name = f"{tc_name}_{idx:03d}.mp4"
        new_path = tc_dir / new_name

        shutil.copy2(original, new_path)

        log_rows.append(
            {
                "old_filename": original_name,
                "new_filename": new_name,
                "test_case": tc_name,
                "status": "copied",
            }
        )

        print(f"✅ {original_name} → {new_name}")
        idx += 1

pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
print(f"\n📄 Log written to {LOG_PATH}")
