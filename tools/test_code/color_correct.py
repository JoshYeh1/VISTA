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

#Other imports 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

vrs_file = "/Users/joshuayeh/dataset_project/VISTA/data/raw/vrs/20250618_objectloc_office.vrs"
output_folder = "/Users/joshuayeh/dataset_project/VISTA/data/processed/"

print(f"Creating data provider from {vrs_file}")
provider = data_provider.create_vrs_data_provider(vrs_file)
if not provider:
    print("Invalid vrs data provider")

# save source image for comparison
stream_id = provider.get_stream_id_from_label("camera-rgb")
provider.set_color_correction(False)
provider.set_devignetting(False) 
src_image_array = provider.get_image_data_by_index(stream_id, 0)[0].to_numpy_array()

provider.set_color_correction(True) 
provider.set_devignetting(False) 
color_corrected_image_array = provider.get_image_data_by_index(stream_id, 0)[0].to_numpy_array()

# visualize input and results
plt.figure()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"Color Correction")

axes[0].imshow(src_image_array, vmin=0, vmax=255)
axes[0].title.set_text(f"before color correction")
axes[1].imshow(color_corrected_image_array, vmin=0, vmax=255)
axes[1].title.set_text(f"after color correction")

plt.show()