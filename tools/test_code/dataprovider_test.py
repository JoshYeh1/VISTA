import sys
import os 

#Adding projectaria_tools repo 
repo_path = os.path.abspath(os.path.join(os.getcwd(),"../../extract_dataset/"))
sys.path.insert(0,repo_path)
print(repo_path)

#Aria SDK imports
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId

#Other imports 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

vrs_file = "/Users/joshuayeh/dataset_project/VISTA/data/raw/vrs/20250618_objectloc_office.vrs"

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

# Same example using Time
rgb_stream_id = StreamId('214-1')
fig, axes = plt.subplots(1, 4, figsize=(12, 5))
fig.suptitle('Retrieving image data using Time')

print(f"Retrieving image data using Time")

time_domain = TimeDomain.DEVICE_TIME  # query data based on device clock 
option = TimeQueryOptions.CLOSEST # gets the frame closest to the given time
start_time = provider.get_first_time_ns(rgb_stream_id, time_domain)

for idx, [stream_name, stream_id] in enumerate(list(stream_mappings.items())):
    image = provider.get_image_data_by_time_ns(stream_id, start_time, time_domain, option)
    axes[idx].imshow(image[0].to_numpy_array(), cmap="gray", vmin=0, vmax=255)
    axes[idx].title.set_text(stream_name)
    axes[idx].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.show()
