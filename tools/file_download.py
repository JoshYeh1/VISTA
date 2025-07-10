import os

destination_folder = "/Users/joshuayeh/raw_data"

vrs_files = [
    "SD_fountain.vrs",
    "SD_monument.vrs",
    "SD_park_shops.vrs",
    "SD_swan_boat.vrs",
    "HD_bench.vrs",
    "TU_no_feeding_CD.vrs",
    "TU_poster.vrs",
    "TU_retail.vrs",
    "TU_enter.vrs",
    "SD_intersection.vrs",
    "TU_brand.vrs",
    "TI_door.vrs",
    "SD_live_music.vrs",
    "SD_graveyard.vrs",
    "SD_round_thing.vrs",
    "OL_trash.vrs",
    "SD_water_fountain_.vrs",
    "TU_maccas.vrs",
    "TU_wigs.vrs",
    "SD_street.vrs",
    "TU_eyebrow.vrs",
    "SD_bike.vrs",
    "TU_tailor.vrs",
    "SD_state_house.vrs",
    "TU_monument_sing.vrs",
    "SD_firefighters.vrs",
    "SD_bird.vrs",
    "HD_sidewalk.vrs",
    "HD_wet_floor.vrs",
    "TI_atm.vrs",
    "OQ_building.vrs",
    "Nav_emergergency.vrs",
    "SD_construction.vrs",
    "OQ_water_bottle.vrs",
    "OQ_paper_plates.vrs",
    "OQ_pillow.vrs",
    "OQ_eggs.vrs",
    "OQ_milk.vrs",
    "OQ_sponge.vrs",
    "OQ_cable.vrs",
    "OQ_first_aid.vrs",
    "OQ_spray_bottle.vrs",
    "OQ_kettle.vrs",
    "OQ_turner.vrs",
    "OQ_turners.vrs"
]

for file in vrs_files:
    source_path = f"/sdcard/recording/{file}"
    command = f"adb pull {source_path} {destination_folder}/"
    print(f"Executing: {command}")
    os.system(command)
