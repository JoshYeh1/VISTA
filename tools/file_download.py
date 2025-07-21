#prgram that downloads raw .vrs files to computer
#change destination_folder and vrs_files as needed
import os

destination_folder = "/Volumes/T7 Shield/vista_data/raw"#/Users/joshuayeh/dataset_project/VISTA/data/raw"

vrs_files = [
    "DE16-_Pond.vrs",
    "DE17_stairs.vrs",
    "HD16_TRIP.vrs",
    "HD17-_light_post.vrs",
    "HD18-puddle.vrs",
    "HD19-_walk_into_.vrs",
    "HD20_trip_.vrs",
    "HD21_construction_.vrs",
    "HD22_obstacle_.vrs",
    "HD23_path.vrs",
    "HD24_sidewalk_.vrs",
    "Nav16-Food_stand_.vrs",
    "Nav18-counter.vrs",
    "Nav19-bench.vrs",
    "Nav20-_ticket_counter_.vrs",
    "Nav21_cross_road_.vrs",
    "Nav24_street_.vrs",
    "Nav_17-_statue.vrs",
    "Nav_23_street.vrs",
    "OQ16-_statue.vrs",
    "OQ17-_van.vrs",
    "OQ18_Pillar.vrs",
    "OQ19_trash_can.vrs",
    "OQ20-cold_brew_.vrs",
    "OQ21-ice_cream_.vrs",
    "OQ22-food_stand.vrs",
    "OQ23_sell.vrs",
    "OQ24_no_Objects_.vrs",
    "OQ25_pot.vrs",
    "OQ26_car.vrs",
    "OQ27_no_of_cars_.vrs"
]

for file in vrs_files:
    source_path = f"/sdcard/recording/{file}"
    command = f"adb pull {source_path} {destination_folder}/"
    print(f"Executing: {command}")
    os.system(command)
