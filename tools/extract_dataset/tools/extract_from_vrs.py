import os, cv2, json, numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write as write_wav
from projectaria_tools.core import data_provider as dp

def extract_from_vrs(vrs_path: str, output_dir: str, test_case_id: str = "TC01_03"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    # ---- Create provider directly from the .vrs file ----
    provider: dp.VrsDataProvider = dp.create_vrs_data_provider(vrs_path)

    # --- images ---
    img_ids = [sid for sid in provider.get_all_streams()
            if provider.get_label_from_stream_id(sid) == "camera-rgb"]

    # --- audio ---
    aud_ids = [sid for sid in provider.get_all_streams()
            if provider.get_label_from_stream_id(sid) == "mic"]

    # --- IMU ---
    motion_ids = [sid for sid in provider.get_all_streams()
                if provider.get_label_from_stream_id(sid) == "imu-right"]


    # --------- annotations meta template ----------
    meta = {
        "test_case": "Object Location",
        "setup_description": "4 objects on a wooden table: phone, keys, glasses, charger",
        "user_query": "Where is my phone?",
        "descriptive_ground_truth": "The phone is on the far right side of the table next to a white charger.",
        "action_ground_truth": "Reach toward the far right corner of the table.",
        "task_type": "object_localization",
        "environment": "indoor",
        "lighting": "natural",
        "distance_to_target": 1.2,
        "measurable_result": "location accuracy and spatial reference",
        "human_score": None
    }

    # --------- RGB frames & timestamps ----------
    annotations, ts_csv_lines = [], ["frame_index,timestamp_ns\n"]
    if img_ids:
        print(f"Extracting images from: {img_ids[0]}")
        for idx, img in enumerate(provider.get_sensor_data_sequence(img_ids[0])):
            fname = f"images/{test_case_id}_{idx:02d}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), img.image)
            ts_csv_lines.append(f"{idx},{img.timestamp_ns}\n")
            annotations.append({"id": f"{test_case_id}_{idx:02d}",
                                "scene_image": fname,
                                "timestamp_ns": img.timestamp_ns, **meta})

    # save timestamps & annotations
    with open(os.path.join(output_dir, "timestamps.csv"), "w") as f: f.writelines(ts_csv_lines)
    with open(os.path.join(output_dir, "annotations.json"), "w") as f: json.dump({"annotations": annotations}, f, indent=2)

    # --------- audio ----------
    if aud_ids:
        print(f"Extracting audio from: {aud_ids[0]}")
        samples = []
        for a in provider.get_sensor_data_sequence(aud_ids[0]):
            samples.extend(a.samples)
        write_wav(os.path.join(output_dir, "audio.wav"), 48000, np.array(samples, dtype=np.int16))

    # --------- IMU ----------
    if motion_ids:
        print(f"Extracting IMU from: {motion_ids[0]}")
        imu_lines = ["timestamp_ns,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n"]
        for m in provider.get_sensor_data_sequence(motion_ids[0]):
            imu_lines.append(f"{m.timestamp_ns},{m.gyro[0]},{m.gyro[1]},{m.gyro[2]},"
                             f"{m.accel[0]},{m.accel[1]},{m.accel[2]}\n")
        with open(os.path.join(output_dir, "imu.csv"), "w") as f: f.writelines(imu_lines)

    print(f"\n Extraction complete in {output_dir}")

if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--vrs", required=True, help=".vrs file path")
    p.add_argument("--output", default="output", help="output directory")
    p.add_argument("--test_case_id", default="TC01_03")
    args = p.parse_args()

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    extract_from_vrs(args.vrs, args.output, args.test_case_id)
