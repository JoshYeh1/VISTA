# VISTA
Visually Impaired Scene & Task Assessment

---

## How to Convert Project Aria VRS Files to 10-Second MP4 Videos with Audio

This guide explains, in simple steps, how to turn your Project Aria `.vrs` files into 10-second MP4 videos with audio for easy viewing and annotation.

### Requirements

- **Python 3.7+**
- **FFmpeg** (for video/audio processing)
- **Extracted VRS data** 

### Step-by-Step Instructions

#### 1. Extract Images and Audio from VRS

First, use the VISTA extraction tool to get images and audio from your `.vrs` file:

```bash
cd VISTA/tools
python3 extract_from_vrs.py
# Follow the prompts to select your .vrs file and output folder
```

This will create a folder (e.g., `TC01_01`) with:
- `images/` (contains extracted images)
- `audio.wav` (contains extracted audio)

#### 2. Convert to a 10-Second MP4 Video with Audio

Use FFmpeg to combine the images and audio into a 10-second MP4. Here’s the command (replace `TC01_01` and the paths as needed):

```bash
ffmpeg -y -framerate 0.5 -i VISTA/data/processed/TC01_object_localization/TC01_01/images/TC01_01_%04d.jpg \
-i VISTA/data/processed/TC01_object_localization/TC01_01/audio.wav \
-vf "tpad=stop_mode=clone:stop_duration=8" \
-c:v libx264 -r 10 -pix_fmt yuv420p -c:a aac -shortest -t 10 \
VISTA/data/processed/TC01_object_localization/TC01_01/TC01_01.mp4
```

- This command makes a 10-second video, showing each image for 2 seconds and holding the last image if needed.
- The audio will be included and trimmed to 10 seconds if longer.

#### 3. Repeat for Other Files (Pseudocode)

To convert all folders in a batch, you can use a simple bash loop:

```bash
for d in VISTA/data/processed/TC01_object_localization/TC01_*; do
  ffmpeg -y -framerate 0.5 -i "$d/images/$(basename $d)_%04d.jpg" \
    -i "$d/audio.wav" \
    -vf "tpad=stop_mode=clone:stop_duration=8" \
    -c:v libx264 -r 10 -pix_fmt yuv420p -c:a aac -shortest -t 10 \
    "$d/$(basename $d).mp4"
done
```

---

### Troubleshooting & Special Cases

- **If your folder (e.g., TC01_02) already contains `images/`, `audio.wav`, and `annotations.json`, you do NOT need to run the extraction script or install Python dependencies.**
- Just run the FFmpeg command below, adjusting the prefix as needed:

```bash
ffmpeg -y -framerate 0.5 -i VISTA/data/processed/TC01_object_localization/TC01_02/images/TC01_02_%04d.jpg \
-i VISTA/data/processed/TC01_object_localization/TC01_02/audio.wav \
-vf "tpad=stop_mode=clone:stop_duration=8" \
-c:v libx264 -r 10 -pix_fmt yuv420p -c:a aac -shortest -t 10 \
VISTA/data/processed/TC01_object_localization/TC01_02/TC01_02.mp4
```

- **If you get a 'file not found' error, check that your images are named as expected (e.g., `TC01_02_0000.jpg`, `TC01_02_0027.jpg`, etc.).**
- If the images use a different naming pattern, adjust the FFmpeg input pattern accordingly.

---

**That’s it!** You now have 10-second MP4 videos with audio, ready for annotation or sharing.

If you have any issues, make sure FFmpeg is installed and your images/audio are extracted correctly.
