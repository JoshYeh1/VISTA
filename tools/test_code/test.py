import sys
import os 
import json
import cv2
#Adding projectaria_tools repo 
repo_path = os.path.abspath(os.path.join(os.getcwd(),"../../extract_dataset/"))
sys.path.insert(0,repo_path)
print(repo_path)

#Aria SDK imports
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId
2
#Other imports 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

vrs_file = "/Users/joshuayeh/dataset_project/VISTA/data/raw/vrs/20250618_objectloc_office.vrs"
output_folder = "/Users/joshuayeh/dataset_project/VISTA/data/processed/"

print(f"Creating data provider from {vrs_file}")
provider = data_provider.create_vrs_data_provider(vrs_file)
if not provider:
    print(f"Invalid vrs dataprovider")

stream_mappings = {
    "camera-slam-left": StreamId("1201-1"),
    "camera-slam-right": StreamId("1201-2"),
    "camera-rgb": StreamId("214-1"),
    "camera-eyetracking": StreamId("211-1")
}

meta = {
    "id": "TC01_03",
    "test_case": "Object Location",
    "setup_description": "4 objects on a wooden table: phone, keys, glasses, charger",
    "user_query": "Where is my phone?",
    "descriptive_ground_truth": "Your phone is infront of you, bewtween the bottle and the pen.",
    "action_ground_truth": "Reach infront of you, between the bottle and the pen.",
    "task_type": "object_localization",
    "environment": "indoor",
    "lighting": "natural",
    "distance_to_target": 0.6096,
    "measurable_result": "location accuracy and spatial reference",
    "human_score": None
}

options = (
    provider.get_default_deliver_queued_options()
) #default options activates all streams and plays back all data in full resolution

#Limit playback to start 0.1 seconds after the beginning of the .vrs file
options.set_truncate_first_device_time_ns(int(1e8))  # 0.1 secs after vrs first timestamp
options.set_truncate_last_device_time_ns(int(1e9))  # 1 sec before vrs last timestamp

options.deactivate_stream_all() # deactivate all sensors

# activate only a subset of sensors
rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")  #activate RGB image stream
provider.set_color_correction(True)
options.activate_stream(rgb_stream_id)
options.set_subsample_rate(rgb_stream_id, 1)  # Use every frame

# Activate IMUs
imu_stream_ids = options.get_stream_ids(RecordableTypeId.SLAM_IMU_DATA)
for stream_id in imu_stream_ids:
    options.activate_stream(stream_id)
    options.set_subsample_rate(stream_id, 10)


iterator = provider.deliver_queued_sensor_data(options)
annotations = []
frame_idx = 0
images_folder = os.path.join(output_folder, "images")
annotations_folder = os.path.join(output_folder,"annotations")
os.makedirs(images_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)

# Extract IMU data from 'imu-left'
imu_stream_id = provider.get_stream_id_from_label("imu-left")

accel_x, accel_y, accel_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []
imu_timestamps = []

for index in range(provider.get_num_data(imu_stream_id)):
    imu_data_obj = provider.get_imu_data_by_index(imu_stream_id, index)
    accel_x.append(imu_data_obj.accel_msec2[0])
    accel_y.append(imu_data_obj.accel_msec2[1])
    accel_z.append(imu_data_obj.accel_msec2[2])
    gyro_x.append(imu_data_obj.gyro_radsec[0])
    gyro_y.append(imu_data_obj.gyro_radsec[1])
    gyro_z.append(imu_data_obj.gyro_radsec[2])
    imu_timestamps.append(imu_data_obj.capture_timestamp_ns)

# Optionally keep this if you're using audio timestamps
audio_timestamps = []
iterator = provider.deliver_queued_sensor_data(options)
for sensor_data in iterator:
    if sensor_data.sensor_data_type().name == "AUDIO":
        audio_timestamps.append(sensor_data.get_time_ns(TimeDomain.DEVICE_TIME))

# Reset iterator to stream image data only
stream_id = provider.get_stream_id_from_label("camera-rgb")
num_images = provider.get_num_data(stream_id)

annotations = []
for frame_idx in range(num_images):
    image_data = provider.get_image_data_by_index(stream_id, frame_idx)
    img = image_data[0].to_numpy_array()
    timestamp_ns = image_data[1].capture_timestamp_ns

    img_filename = f"{meta['id']}_{frame_idx:02d}.jpg" #Save Image
    cv2.imwrite(os.path.join(images_folder, img_filename), img)
    print(f"Saved frame {frame_idx}: {img_filename} | timestamp: {timestamp_ns}")

    #find closest imu data
    closest_idx = min(range(len(imu_timestamps)), key=lambda i: abs(imu_timestamps[i] - timestamp_ns))
    closest_imu = {
    "timestamp_ns": imu_timestamps[closest_idx],
    "accel": [accel_x[closest_idx], accel_y[closest_idx], accel_z[closest_idx]],
    "gyro": [gyro_x[closest_idx], gyro_y[closest_idx], gyro_z[closest_idx]]
    }

    #find closest audio timestamp
    closest_audio_ts = min(audio_timestamps, key=lambda x: abs(x - timestamp_ns)) if audio_timestamps else None

    annotation = {
        "scene_image": f"images/{img_filename}",
        "timestamp_ns": timestamp_ns,
        "imu": closest_imu,
        "audio_timestamp_ns": closest_audio_ts,
        **meta
    }

    annotations.append(annotation)
    frame_idx += 1

with open(os.path.join(annotations_folder, "annotations.json"), "w") as f:
    json.dump({"annotations": annotations}, f, indent=2)
