from pathlib import Path
from collections import Counter

folder = Path("/Volumes/T7 Shield/json_stuff/annotator_agreement")

counts = Counter()

for f in folder.glob("*.json"):
    clip = "_".join(f.stem.split("_")[:2])
    counts[clip] += 1

for clip, n in sorted(counts.items()):
    print(clip, n)