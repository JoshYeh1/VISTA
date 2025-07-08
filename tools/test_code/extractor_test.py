import os, json, csv, wave, argparse, pathlib, cv2, numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId

# -------------------- CLI --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--vrs", default="/Users/joshuayeh/dataset_project/VISTA/data/raw/20250618_objectloc_office.vrs")
parser.add_argument("--output", default="/Users/joshuayeh/dataset_project/VISTA/data/processed/TC01_object_localization/TC01_00/")
parser.add_argument("--test_case_id", default="TC01_00")
parser.add_argument("--test_type", default="Object Location")
parser.add_argument("--key_frames", type=int, default=5,
                    help="How many RGB frames to annotate")
args = parser.parse_args()
out_path = pathlib.Path(args.output)
images_folder = out_path / "images"
annotations_folder = out_path
images_folder.mkdir(parents=True, exist_ok=True)
annotations_folder.mkdir(exist_ok=True)

# ----------------- Open provider -------------
provider = data_provider.create_vrs_data_provider(args.vrs)
if provider is None:
    raise RuntimeError("Invalid VRS file")

# ------------ AUDIO to WAV -------------------
mic_sid = provider.get_stream_id_from_label("mic")
n_audio = provider.get_num_data(mic_sid)

output_wav = annotations_folder / "audio.wav"
sample_rate = 48000
n_channels = 7  # change this if your setup uses a different count

# Load audio blocks
samples = []
for i in range(n_audio):
    audio_data = provider.get_audio_data_by_index(mic_sid, i)
    raw_block = audio_data[0].data
    samples.append(np.asarray(raw_block, dtype=np.int32))

# Flatten and reshape
flat_audio = np.concatenate(samples)
assert len(flat_audio) % n_channels == 0, "Audio length not divisible by number of channels"
multi_audio = flat_audio.reshape(-1, n_channels)

# Normalize to 16-bit PCM
max_val = np.max(np.abs(multi_audio))
pcm16 = ((multi_audio / max_val) * 32767).astype(np.int16)

# Save to .wav
with wave.open(str(annotations_folder / output_wav), "wb") as wf:
    wf.setnchannels(n_channels)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sample_rate)
    wf.writeframes(pcm16.tobytes())

print(f"Wrote multichannel audio to: {output_wav} ({n_channels} ch, {sample_rate} Hz)")

# --------------- IMU to CSV ------------------
imu_sid   = provider.get_stream_id_from_label("imu-left")
n_imu     = provider.get_num_data(imu_sid)
imu_csv   = annotations_folder / "imu_left.csv"
with open(imu_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ns",
                     "accel_x","accel_y","accel_z",
                     "gyro_x","gyro_y","gyro_z"])
    for i in range(n_imu):
        d = provider.get_imu_data_by_index(imu_sid, i)
        writer.writerow([d.capture_timestamp_ns,
                         *d.accel_msec2, *d.gyro_radsec])
print(f"Wrote {imu_csv}  ({n_imu} rows)")

# ------------- RGB key-frames ---------------
rgb_sid    = provider.get_stream_id_from_label("camera-rgb")
n_images   = provider.get_num_data(rgb_sid)
indices    = np.linspace(0, n_images-1, args.key_frames, dtype=int)

# common meta, shortened here for clarity
meta = {
    "id": f"{args.test_case_id}",
    "dataset_version": 1.0,
    "collection_date": "2025-06-18",
    "fps": 10,
    "camera_resolution": [1408, 1408],
    "test_case": args.test_type,
    "setup_description": "4 objects on a table: plastic bottle, phone, leather journal, headphones, pen",
    "user_query": "Where is my phone?",
    "descriptive_ground_truth": "Your phone is in front of you, between the bottle and the pen.",
    "action_ground_truth": "Reach in front of you, between the bottle and the pen.",
    "task_type": "object_localization",
    "environment": "indoor",
    "lighting": "fluorescent",
    "distance_to_target": 0.6096,
    "model_description_output": None,
    "model_action_output": None,
    "description_score": None,
    "action_score": None,
    "evaluation_method": None,
    "measurable_result": "location accuracy and spatial reference",
    "human_score": {
        "description_accuracy": None,
        "action_usefulness": None,
        "confidence": None
    }
}

# Preload all IMU timestamps so we can look up closest match later
imu_timestamps = []
for i in range(n_imu):
    d = provider.get_imu_data_by_index(imu_sid, i)
    imu_timestamps.append(d.capture_timestamp_ns)


annotations = []
for idx in indices:
    img_data = provider.get_image_data_by_index(rgb_sid, idx)
    img      = cv2.cvtColor(np.rot90(img_data[0].to_numpy_array(), 3),
                            cv2.COLOR_RGB2BGR)
    ts_ns    = img_data[1].capture_timestamp_ns
    fname    = f"{meta['id']}_{idx:04d}.jpg"
    cv2.imwrite(str(images_folder / fname), img)

    # Find the closest IMU index based on timestamp
    imu_idx = min(range(n_imu), key=lambda i: abs(imu_timestamps[i] - ts_ns))

    annotation = {
        "scene_image": f"images/{fname}",
        "timestamp_ns": ts_ns,
        "imu_row": imu_idx, # pointer into imu_left.csv
        "audio_file": str(output_wav.relative_to(out_path)),
        **meta
    }
    annotations.append(annotation)
    print(f"Saved key-frame {idx} ({fname})")

with open(annotations_folder / "annotations.json", "w") as f:
    json.dump({"annotations": annotations}, f, indent=2)

print("Done")

