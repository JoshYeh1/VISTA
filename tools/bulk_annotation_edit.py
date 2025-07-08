import json

json_path = "/Users/joshuayeh/dataset_project/VISTA/data/processed/TC01_object_localization/TC01_01/annotations.json"

#fields you want to apply to all annotations
shared_fields = [
    "collection_date",
    "test_case",
    "setup_description",
    "user_query",
    "descriptive_ground_truth",
    "action_ground_truth",
    "task_type",
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

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("Updates applied to all annotations.")
