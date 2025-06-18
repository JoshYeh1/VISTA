import cv2
import os
import json
from projectaria_tools import VRSReader

# === CONFIG ===
VRS_DIR = "vrs_raw" #input videos folder
IMAGE_DIR = "images" #input images folder
ANNOTATION_FILE = "annotations/object_location.jsonl"
FRAME_INTERVAL_SECONDS = 1  #extracts one frame per second

# Ensure image and annotation folders exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ANNOTATION_FILE), exist_ok=True)

# reads file, finds frames to extract and labels them

def extract_frames_and_annotate(vrs_path, frame_interval_s, video_id):
    reader = VRSReader(vrs_path)
    rgb_stream_id = reader.get_streams_by_type("camera-rgb")[0]

    last_saved_ns = 0
    interval_ns = int(frame_interval_s * 1e9)
    frame_index = 0
    annotations = []

    for data in reader.get_stream_data(rgb_stream_id):
        ts = data.timestamp_ns
        if ts - last_saved_ns >= interval_ns:
            image = data.image_data_to_cv_mat()
            image_name = f"{video_id}_frame_{frame_index:03d}.jpg"
            image_path = os.path.join(IMAGE_DIR, image_name)
            cv2.imwrite(image_path, image)

            annotation = {
                "id": f"{video_id}_F{frame_index}",
                "video_id": video_id,
                "frame_index": frame_index,
                "scene_image": image_path,
                "user_query": "Where is my phone?",
                "descriptive_ground_truth": "",
                "action_ground_truth": "",
                "task_type": "object_localization",
                "environment": "indoor",
                "lighting": "unknown",
                "measurable_result": "Object location accuracy and spatial reference",
                "future_fields": {
                    "timestamp": round(ts / 1e9, 2),
                    "video_path": vrs_path
                }
            }

            annotations.append(annotation)
            last_saved_ns = ts
            frame_index += 1

    return annotations

# === MAIN ===
all_annotations = []
for filename in os.listdir(VRS_DIR):
    if filename.endswith(".vrs"):
        vrs_path = os.path.join(VRS_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        print(f"Processing {video_id}...")
        annotations = extract_frames_and_annotate(vrs_path, FRAME_INTERVAL_SECONDS, video_id)
        all_annotations.extend(annotations)

# Save annotations
with open(ANNOTATION_FILE, "w") as f:
    for ann in all_annotations:
        f.write(json.dumps(ann) + "\n")

print(f"Saved {len(all_annotations)} annotations to {ANNOTATION_FILE}")