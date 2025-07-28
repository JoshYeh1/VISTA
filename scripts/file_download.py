#prgram that downloads raw .vrs files to computer
#change destination_folder and vrs_files as needed
import os

destination_folder = "/Users/joshuayeh/dataset_project/data/test"

vrs_files = ["TU-director_.vrs","OQ-Tissue_.vrs"]


for file in vrs_files:
    source_path = f"/sdcard/recording/{file}"
    command = f"adb pull {source_path} {destination_folder}/"
    print(f"Executing: {command}")
    os.system(command)
