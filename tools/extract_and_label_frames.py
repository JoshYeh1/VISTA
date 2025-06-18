import cv2
import os
import json

# === CONFIG ===
VIDEO_DIR = "videos"
IMAGE_DIR = "images"
ANNOTATION_FILE = "annotations/object_location.jsonl"
FRAME_INTERVAL_SECONDS = 1  # extract one frame per second

# Ensure image and annotation folders exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ANNOTATION_FILE), exist_ok=True)

# === FUNCTION TO EXTRACT FRAMES AND GENERATE ANNOTATION ENTRIES ===
def extract_frames_and_label(video_path, frame_interval, video_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_frames = int(fps * frame_interval)

    count = 0
    frame_index = 0
    annotations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval_frames == 0:
            image_name = f"{video_id}_frame_{frame_index:03d}.jpg"
            image_path = os.path.join(IMAGE_DIR, image_name)
            cv2.imwrite(image_path, frame)

            # Create a starter annotation with placeholders
            annotation = {
                "id": f"{video_id}_F{frame_index}",
                "video_id": video_id,
                "frame_index": frame_index,
                "scene_image": image_path,
                "user_query": "Where is my phone?",
                "descriptive_ground_truth": "",  # Fill this in later
                "action_ground_truth": "",        # Fill this in later
                "task_type": "object_localization",
                "environment": "indoor",
                "lighting": "unknown",
                "measurable_result": "Object location accuracy and spatial reference",
                "future_fields": {
                    "timestamp": round(count / fps, 2),
                    "video_path": video_path
                }
            }
            annotations.append(annotation)
            frame_index += 1

        count += 1

    cap.release()
    return annotations

# === MAIN LOOP ===
all_annotations = []
for filename in os.listdir(VIDEO_DIR):
    if filename.lower().endswith((".mp4", ".mov")):
        video_path = os.path.join(VIDEO_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        print(f"Processing {video_id}...")
        annotations = extract_frames_and_label(video_path, FRAME_INTERVAL_SECONDS, video_id)
        all_annotations.extend(annotations)

# Save annotations as JSONL
with open(ANNOTATION_FILE, "w") as f:
    for entry in all_annotations:
        f.write(json.dumps(entry) + "\n")

print(f"Extracted frames saved to `{IMAGE_DIR}` and annotations to `{ANNOTATION_FILE}`")
