import json
import logging
from pathlib import Path
from collections import defaultdict

# ============================
# CONFIG
# ============================

INPUT_FILE = Path("/Volumes/T7 Shield/json_files/ryan_1.json")

MAIN_OUTPUT_DIR = Path("/Volumes/T7 Shield/json_stuff/renamed_json")
OVERLAP_DIR = Path("/Volumes/T7 Shield/json_stuff/overlapping_annotations")
DUPLICATE_DIR = Path("/Volumes/T7 Shield/json_stuff/duplicate_annotations")

MAIN_OUTPUT_DIR.mkdir(exist_ok=True)
OVERLAP_DIR.mkdir(exist_ok=True)
DUPLICATE_DIR.mkdir(exist_ok=True)

# ============================
# LOGGING
# ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MAIN_OUTPUT_DIR / "pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ============================
# COUNTERS
# ============================

duplicate_counters: dict[str, int] = defaultdict(int)
overlap_counters: dict[str, int] = defaultdict(int)

# ============================
# LOAD EXPORT
# ============================

with open(INPUT_FILE, "r") as input_file:
    tasks = json.load(input_file)

log.info(f"Loaded {len(tasks)} tasks from {INPUT_FILE}")

# ============================
# HELPERS
# ============================

def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as out_file:
        json.dump(data, out_file, indent=2)


def parse_annotation(annotation: dict) -> dict:

    annotation_struct = {
        "annotation_id": annotation.get("id"),
        "annotator_id": annotation.get("completed_by"),
        "unique_id": annotation.get("unique_id"),
        "created_at": annotation.get("created_at"),
        "lead_time_seconds": annotation.get("lead_time"),
        "was_cancelled": annotation.get("was_cancelled"),
        "ground_truth": annotation.get("ground_truth"),
        "video_level": {},
        "temporal_segments": [],
    }

    goals: dict[str, str] = {}
    instructions: dict[str, str] = {}
    segments: dict[tuple, dict] = defaultdict(dict)

    for item in annotation.get("result", []):
        from_name = item.get("from_name")
        value = item.get("value", {})

        # ---------- VIDEO LEVEL ----------
        if "ranges" not in value:
            text_list = value.get("text")
            text = text_list[0] if text_list else None

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

        # ---------- TEMPORAL ----------
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
                seg_text = value["text"][0]
                if from_name == "user_question_seg":
                    segments[key]["user_question"] = seg_text
                elif from_name == "scene_description_seg":
                    segments[key]["scene_description"] = seg_text
                elif from_name == "answer_seg":
                    segments[key]["answer"] = seg_text
                elif from_name == "action_seg":
                    segments[key]["action"] = seg_text

    actions = []
    all_keys = sorted(set(goals) | set(instructions), key=lambda x: int(x))
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

    return annotation_struct


# ============================
# MAIN LOOP
# ============================

stats = {"saved": 0, "overlaps": 0, "duplicates": 0, "skipped": 0}

for task in tasks:

    video_file = task.get("file_upload")
    if not video_file:
        video_file = task.get("data", {}).get("video")

    if not video_file:
        log.warning(f"Task {task.get('id')} has no video path — skipping.")
        stats["skipped"] += 1
        continue

    filename = video_file.split("/")[-1]
    base = filename.replace("_blurred.mp4", "")
    clip_id = base.split("-")[-1]  # ✅ FIXED: remove hash prefix

    for annotation in task.get("annotations", []):
        try:
            annotation_struct = parse_annotation(annotation)
        except Exception as exc:
            log.error(
                f"Failed to parse annotation {annotation.get('id')} "
                f"for clip {clip_id}: {exc}"
            )
            stats["skipped"] += 1
            continue

        output_path = MAIN_OUTPUT_DIR / f"{clip_id}.json"

        try:
            if output_path.exists():
                with open(output_path, "r") as existing_file:
                    existing_data = json.load(existing_file)

                existing_unique_ids = {
                    ann.get("unique_id")
                    for ann in existing_data.get("annotations", [])
                }

                if annotation_struct["unique_id"] in existing_unique_ids:
                    duplicate_counters[clip_id] += 1
                    dup_path = DUPLICATE_DIR / f"{clip_id}_duplicate_{duplicate_counters[clip_id]}.json"
                    save_json(dup_path, annotation_struct)
                    stats["duplicates"] += 1
                    continue

                else:
                    existing_data["annotations"].append(annotation_struct)
                    save_json(output_path, existing_data)

                    overlap_counters[clip_id] += 1
                    overlap_path = OVERLAP_DIR / f"{clip_id}_overlap_{overlap_counters[clip_id]}.json"
                    save_json(overlap_path, annotation_struct)
                    stats["overlaps"] += 1

            else:
                clip_data = {
                    "clip_id": clip_id,
                    "video_filename": video_file,
                    "task_id": task.get("id"),
                    "annotations": [annotation_struct],
                }
                save_json(output_path, clip_data)
                stats["saved"] += 1

        except Exception as exc:
            log.error(f"Failed to write files for clip {clip_id}: {exc}")
            stats["skipped"] += 1

# ============================
# SUMMARY
# ============================

log.info("━" * 50)
log.info("Pipeline complete.")
log.info(f"  New clips saved  : {stats['saved']}")
log.info(f"  Overlaps logged  : {stats['overlaps']}")
log.info(f"  Duplicates logged: {stats['duplicates']}")
log.info(f"  Skipped (errors) : {stats['skipped']}")
log.info("━" * 50)