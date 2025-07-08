import json
from pathlib import Path

out_path = Path("/Users/joshuayeh/dataset_project/VISTA/data/processed/")
case_num = int(input("Input Test Case number (1-10): "))
specific_number = input("Input Id number (00-99): ")

if case_num == 1:
    test_case_id= "TC01_" + specific_number
    annotations_folder = out_path / f"TC01_object_localization/{test_case_id}"
if case_num == 2:
    test_case_id = "TC02_" + specific_number
    annotations_folder = out_path / f"TC02_hzd_detection/{test_case_id}"
if case_num == 3:
    test_case_id = "TC03_" + specific_number
    annotations_folder = out_path / f"TC03_scene_description/{test_case_id}"
if case_num == 4:
    test_case_id = "TC04_" + specific_number
    annotations_folder = out_path / f"TC04_navigation/{test_case_id}"
if case_num == 5:
    test_case_id = "TC05_" + specific_number
    annotations_folder = out_path / f"TC05_social_cues/{test_case_id}"
if case_num == 6:
    test_case_id = "TC06_" + specific_number
    annotations_folder = out_path / f"TC06_distance_est/{test_case_id}"
if case_num == 7:
    test_case_id = "TC07_" + specific_number
    annotations_folder = out_path / f"TC07_task_instruction/{test_case_id}"
if case_num == 8:
    test_case_id = "TC08_" + specific_number
    annotations_folder = out_path / f"TC08_object_query/{test_case_id}"
if case_num == 9:
    test_case_id = "TC09_" + specific_number
    annotations_folder = out_path / f"TC09_txt_understanding/{test_case_id}"
if case_num == 10:
    test_case_id = "TC10_" + specific_number
    annotations_folder = out_path / f"TC10_motion_understanding/{test_case_id}"
if case_num > 10:
    print(f"Invalid Case Number")
    exit

json_path = annotations_folder / "annotations.json"

#fields you want to apply to all annotations
shared_fields = [
    "collection_date",
    "setup_description",
    "user_query",
    "descriptive_ground_truth",
    "action_ground_truth",
    "environment",
    "lighting",
    "distance_to_target",
    "measurable_result"
]

with open(json_path, "r") as f:
    data = json.load(f)

annotations = data["annotations"]
updates = {}
print("Enter new values to apply to ALL annotations (press Enter to skip):")
for field in shared_fields:
    value = input(f"{field}: ").strip()
    if value:
        if field == "distance_to_target":
            try:
                value = float(value)
            except ValueError:
                print("Invalid float for distance_to_target. Skipping.")
                continue
        updates[field] = value

for ann in annotations:
    for key, value in updates.items():
        ann[key] = value

with open(str(json_path), "w") as f:
    json.dump(data, f, indent=2)

print("Updates applied to all annotations.")
