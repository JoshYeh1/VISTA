import pandas as pd
import shutil
import subprocess
from pathlib import Path

CSV_PATH = Path("/Users/joshuayeh/Documents/VISTA Dataset Listings - Collected Data.csv")
RAW_DIR  = Path("/Volumes/T7 Shield/all_mp4_video")
OUT_DIR  = Path("/Volumes/T7 Shield/VISTA_clean")
LOG_PATH = Path("/Volumes/T7 Shield/rename_log.csv")

# Audio extraction settings
EXTRACT_WAV = True          # set False if you want to skip extraction
WAV_SR = 16000              # common for ASR; change to 48000 if you prefer
WAV_MONO = True             # True -> -ac 1, False -> keep channels
REQUIRE_FFMPEG = True       # if True, fails clearly if ffmpeg/ffprobe missing

OUT_DIR.mkdir(parents=True, exist_ok=True)

def _check_tool_exists(name: str) -> bool:
    try:
        subprocess.run([name, "-version"], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

if EXTRACT_WAV and REQUIRE_FFMPEG:
    if not _check_tool_exists("ffmpeg") or not _check_tool_exists("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install with: brew install ffmpeg"
        )

def has_audio_stream(mp4_path: Path) -> bool:
    """
    Returns True if ffprobe finds at least one audio stream.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(mp4_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""

def extract_wav(mp4_path: Path, wav_path: Path, sr: int = 16000, mono: bool = True):
    """
    Extract audio as PCM 16-bit WAV. Optionally resample and downmix to mono.
    """
    cmd = ["ffmpeg", "-y", "-i", str(mp4_path), "-vn"]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sr), "-acodec", "pcm_s16le", str(wav_path)]
    subprocess.run(cmd, check=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["File Name", "Case ID No"])
df = df.sort_values(by=["Case ID No", "File Name"])

log_rows = []

for case_id, group in df.groupby("Case ID No"):
    tc_name = f"TC{int(float(case_id)):02d}"
    tc_dir = OUT_DIR / tc_name
    tc_dir.mkdir(parents=True, exist_ok=True)

    idx = 1
    for _, row in group.iterrows():
        original_name = Path(row["File Name"]).stem + ".mp4"
        original = RAW_DIR / original_name

        if not original.exists():
            print(f"⚠️ Missing MP4: {original_name}")
            log_rows.append({
                "old_filename": original_name,
                "new_basename": "",
                "new_mp4_path": "",
                "new_wav_path": "",
                "test_case": tc_name,
                "status": "missing"
            })
            continue

        base_id = f"{tc_name}_{idx:03d}"

        # folder per recording inside the test case folder
        rec_dir = tc_dir / base_id
        rec_dir.mkdir(parents=True, exist_ok=True)

        # copy MP4 into that folder, renamed
        new_mp4_path = rec_dir / f"{base_id}.mp4"
        shutil.copy2(original, new_mp4_path)

        # placeholders for future artifacts
        json_path = rec_dir / f"{base_id}.json"
        vrs_path  = rec_dir / f"{base_id}.vrs"
        json_path.touch(exist_ok=True)
        vrs_path.touch(exist_ok=True)

        # WAV path (we will create it by extraction if possible)
        wav_path = rec_dir / f"{base_id}.wav"

        wav_status = "skipped"
        if EXTRACT_WAV:
            try:
                if has_audio_stream(new_mp4_path):
                    extract_wav(new_mp4_path, wav_path, sr=WAV_SR, mono=WAV_MONO)
                    wav_status = "extracted"
                else:
                    # ensure we don't leave a misleading empty wav file
                    if wav_path.exists():
                        wav_path.unlink()
                    wav_status = "no_audio_stream"
            except subprocess.CalledProcessError as e:
                # keep going, but log failure
                if wav_path.exists() and wav_path.stat().st_size == 0:
                    wav_path.unlink()
                wav_status = f"ffmpeg_failed"

        status = f"copied+folder+placeholders+wav_{wav_status}"

        log_rows.append({
            "old_filename": original_name,
            "new_basename": base_id,
            "new_mp4_path": str(new_mp4_path),
            "new_wav_path": str(wav_path) if wav_path.exists() else "",
            "test_case": tc_name,
            "status": status
        })

        print(f"✅ {original_name} → {new_mp4_path} | WAV: {wav_status}")
        idx += 1

pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
print(f"\n📄 Log written to {LOG_PATH}")
