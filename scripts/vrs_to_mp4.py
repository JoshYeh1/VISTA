#This program takes all the vrs files in one folder and converts them into mp4 files

import os
from pathlib import Path
from projectaria_tools.utils.vrs_to_mp4_utils import convert_vrs_to_mp4

input_folder = Path("/Users/joshuayeh/dataset_project/data/new_raw") #folder with vrs files
output_folder = Path("/Users/joshuayeh/dataset_project/hugging_face/VISTA/raw") #folder for mp4 videos
log_folder = Path("/Users/joshuayeh/dataset_project/hugging_face/VISTA/logs")#folder for timestamp logs
down_sample_factor = 1
output_folder.mkdir(parents=True, exist_ok=True)
log_folder.mkdir(parents=True, exist_ok=True)

for vrs_file in input_folder.glob("*.vrs"):
    output_mp4 = output_folder / (vrs_file.stem + ".mp4")
    print(f"Converting: {vrs_file.name} -> {output_mp4.name}")
    try:
        convert_vrs_to_mp4(str(vrs_file), str(output_mp4), str(log_folder), down_sample_factor)
    except Exception as e:
        print(f"Failed to convert {vrs_file.name}: {e}")
