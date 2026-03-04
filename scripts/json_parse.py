import json
from pathlib import Path
from collections import defaultdict

# ============================
# CONFIG
# ============================

INPUT_FILE = Path("/Volumes/T7 Shield/json_files/will_2.json")

MAIN_OUTPUT_DIR = Path("/Volumes/T7 Shield/renamed_json")
REPEAT_OUTPUT_DIR = Path("/Volumes/T7 Shield/repeated_annotations")

MAIN_OUTPUT_DIR.mkdir(exist_ok=True)
REPEAT_OUTPUT_DIR.mkdir(exist_ok=True)

# ============================
# LOAD EXPORT
# ============================

with open(INPUT_FILE, "r") as f:
    tasks = json.load(f)

# ============================
# MAIN LOOP
# ============================

for task in tasks:

    video_file = task.get("file_upload")
    if not video_file:
        continue

    clip_id = video_file.split("-")[-1].replace("_blurred.mp4", "")

    for annotation in task.get("annotations", []):

        annotation_struct = {
            "annotation_id": annotation.get("id"),
            "annotator_id": annotation.get("completed_by"),
            "unique_id": annotation.get("unique_id"),
            "created_at": annotation.get("created_at"),
            "lead_time_seconds": annotation.get("lead_time"),
            "was_cancelled": annotation.get("was_cancelled"),
            "ground_truth": annotation.get("ground_truth"),
            "video_level": {},
            "temporal_segments": []
        }

        goals = {}
        instructions = {}
        segments = defaultdict(dict)

        # ============================
        # PARSE RESULT ITEMS
        # ============================

        for item in annotation.get("result", []):

            from_name = item.get("from_name")
            value = item.get("value", {})

            # ----------------------------
            # VIDEO-LEVEL ANNOTATIONS
            # ----------------------------

            if "ranges" not in value:

                text = value.get("text")
                if text:
                    text = text[0]

                if from_name == "user_question_whole":
                    annotation_struct["video_level"]["question"] = text

                elif from_name == "scene_description_whole":
                    annotation_struct["video_level"]["scene_description"] = text

                elif from_name == "answer_whole":
                    annotation_struct["video_level"]["answer"] = text

                elif from_name and "action_goal_" in from_name:
                    idx = from_name.split("_")[-1]
                    goals[idx] = text

                elif from_name and "action_instruction_" in from_name:
                    idx = from_name.split("_")[-1]
                    instructions[idx] = text

            # ----------------------------
            # TEMPORAL SEGMENTS
            # ----------------------------

            else:
                ranges = value.get("ranges")
                if not ranges:
                    continue

                start = ranges[0]["start"]
                end = ranges[0]["end"]
                key = (start, end)

                segments[key]["start_frame"] = start
                segments[key]["end_frame"] = end

                if "timelinelabels" in value:
                    segments[key]["timeline_labels"] = value["timelinelabels"]

                if "choices" in value:
                    segments[key]["linked_goal"] = value["choices"]

                if "text" in value:
                    text = value["text"][0]

                    if from_name == "user_question_seg":
                        segments[key]["user_question"] = text

                    elif from_name == "scene_description_seg":
                        segments[key]["scene_description"] = text

                    elif from_name == "answer_seg":
                        segments[key]["answer"] = text

                    elif from_name == "action_seg":
                        segments[key]["action"] = text

        # ============================
        # BUILD VIDEO-LEVEL ACTIONS
        # ============================

        actions = []
        all_keys = sorted(
            set(goals.keys()) | set(instructions.keys()),
            key=lambda x: int(x)
        )

        for k in all_keys:
            entry = {}
            if k in goals:
                entry["goal"] = goals[k]
            if k in instructions:
                entry["instruction"] = instructions[k]
            actions.append(entry)

        if actions:
            annotation_struct["video_level"]["actions"] = actions

        annotation_struct["temporal_segments"] = list(segments.values())

        # ============================
        # SAFE SAVE LOGIC
        # ============================

        output_path = MAIN_OUTPUT_DIR / f"{clip_id}.json"

        # If clip file already exists
        if output_path.exists():

            with open(output_path, "r") as f:
                existing_data = json.load(f)

            existing_annotations = existing_data.get("annotations", [])

            existing_unique_ids = {
                ann.get("unique_id") for ann in existing_annotations
            }

            if annotation_struct.get("unique_id") in existing_unique_ids:
                # Duplicate found → save separately
                repeat_path = REPEAT_OUTPUT_DIR / f"{clip_id}_ann{annotation_struct['annotation_id']}_duplicate.json"
                with open(repeat_path, "w") as f:
                    json.dump(annotation_struct, f, indent=2)

            else:
                # Legit overlapping annotation → append
                existing_data["annotations"].append(annotation_struct)

                with open(output_path, "w") as f:
                    json.dump(existing_data, f, indent=2)

        else:
            # First time seeing this clip
            clip_data = {
                "clip_id": clip_id,
                "video_filename": video_file,
                "task_id": task.get("id"),
                "annotations": [annotation_struct]
            }

            with open(output_path, "w") as f:
                json.dump(clip_data, f, indent=2)

print("✅ Canonical VISTA dataset built successfully with duplicate protection.")