#prgram that downloads raw .vrs files to computer
#change destination_folder and vrs_files as needed
import os

destination_folder = "/Users/joshuayeh/dataset_project/hugging_face/VISTA/new_raw"

vrs_files = [
    "OQ_mandarinsoda.vrs",
    "TU_doorsopen.vrs",
    "TU_noaccess.vrs",
    "TU_museumhours.vrs",
    "TU_constructionsite.vrs",
    "TU_gateE.vrs",
    "TU_TempFDC.vrs",
    "TU_flour.vrs",
    "TU_gongcha.vrs",
    "TI_ordering.vrs",
    "OQ_shavingcream.vrs",
    "OQ_shavingcream_2.vrs",
    "TU_aisle4.vrs",
    "TU_aisle14.vrs",
    "TU_kidscards.vrs",
    "OQ_grandmacard.vrs",
    "TI_lift.vrs",
    "TU_doughnuts.vrs",
    "OQ_straws.vrs",
    "TU_oatright.vrs",
    "TU_streetsign.vrs",
    "TU_bite.vrs"
]


for file in vrs_files:
    source_path = f"/sdcard/recording/{file}"
    command = f"adb pull {source_path} {destination_folder}/"
    print(f"Executing: {command}")
    os.system(command)
