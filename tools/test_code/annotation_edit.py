import json
import os
import cv2

# === Config ===
json_path = "/Users/joshuayeh/dataset_project/VISTA/data/processed/TC01_object_localization/TC01_01/annotations.json"  # Path to your JSON metadata
image_root = "/Users/joshuayeh/dataset_project/VISTA/data/processed/TC01_object_localization/TC01_01/"  # Root folder containing the 'images/' folder

# Fields to edit
fields = [
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

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

annotations = data["annotations"]

for i, ann in enumerate(annotations):
    image_path = os.path.join(image_root, ann["scene_image"])

    # Load and display image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Could not open image: {image_path}")
        continue

    window_name = f"{i+1}/{len(annotations)} - {os.path.basename(image_path)}"
    cv2.imshow(window_name, image)

    print(f"\n🖊️ Editing Annotation {i+1}/{len(annotations)}")
    print(f"Scene: {ann['scene_image']}")
    
    for field in fields:
        old_val = ann.get(field)
        prompt = f"{field} [{old_val}]: "
        new_val = input(prompt).strip()

        if new_val:
            if field == "distance_to_target":
                try:
                    ann[field] = float(new_val)
                except ValueError:
                    print("Invalid float. Keeping original.")
            else:
                ann[field] = new_val

    # Save updated file after each annotation
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    cv2.destroyAllWindows()
    proceed = input("Next? (Enter to continue, q to quit): ").strip().lower()
    if proceed == "q":
        break
