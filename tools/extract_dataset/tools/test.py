# helper to see id → label pairs
from projectaria_tools.core import data_provider as dp
prov = dp.create_vrs_data_provider("../../../data/raw/vrs/20250618_objectloc_office.vrs")
for sid in prov.get_all_streams():
    print(sid, prov.get_label_from_stream_id(sid))   # <-- prints '214-1 camera-rgb' etc.
