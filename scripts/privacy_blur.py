import subprocess
from pathlib import Path

# ====== CONFIG ======
OUT_DIR = Path("/Volumes/T7 Shield/VISTA_clean")

EGOBLUR_BIN = "egoblur-gen2"  # assumes it's on PATH (pip install egoblur)
CAMERA_NAME = "camera-rgb"

FACE_MODEL = Path("/Users/joshuayeh/Documents/models/egoblur/ego_blur_face_gen2.jit")
LP_MODEL   = Path("/Users/joshuayeh/Documents/models/egoblur/ego_blur_lp_gen2.jit")

SCALE_FACTOR = "1.15"

# Put the test cases where you DO NOT want face blurring (faces are the point)
# Example: {"TC05", "TC12"} etc.
SOCIAL_CUES_TCS = {
    # "TC07",
    # "TC11",
}

# If True: create a blurred MP4 that includes audio (re-mux from original)
REMUX_AUDIO = True

# If True: skip if blurred output already exists
SKIP_IF_EXISTS = True
# ====================


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def build_egoblur_cmd(input_mp4: Path, output_mp4: Path, blur_faces: bool) -> list[str]:
    """
    Build egoblur-gen2 command.
    For lp-only (social cues), omit face model entirely.
    """
    cmd = [
        EGOBLUR_BIN,
        "--camera_name", CAMERA_NAME,
        "--lp_model_path", str(LP_MODEL),
        "--scale_factor_detections", SCALE_FACTOR,
        "--input_video_path", str(input_mp4),
        "--output_video_path", str(output_mp4),
    ]
    if blur_faces:
        # Add face model only when we want face blurring
        cmd.insert(4, "--face_model_path")
        cmd.insert(5, str(FACE_MODEL))
    return cmd

def remux_audio_from_original(original_mp4: Path, video_only_mp4: Path, out_mp4: Path) -> None:
    """
    EgoBlur output is typically video-only; this copies audio from the original MP4 if present.
    If the original has no audio stream, ffmpeg will fail; we catch that at caller.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only_mp4),
        "-i", str(original_mp4),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_mp4),
    ]
    run(cmd)

def main():
    if not LP_MODEL.exists():
        raise FileNotFoundError(f"LP model not found: {LP_MODEL}")
    if not FACE_MODEL.exists():
        # Only required when blur_faces=True; still warn early so you notice.
        print(f"⚠️ Face model not found (face blurring will fail if enabled): {FACE_MODEL}")

    mp4s = sorted(OUT_DIR.rglob("*.mp4"))
    print(f"Found {len(mp4s)} MP4s under {OUT_DIR}")

    ok = 0
    skipped = 0
    failed = 0

    for mp4 in mp4s:
        # Expect path .../TCxx/TCxx_yyy/TCxx_yyy.mp4
        tc_name = mp4.parents[1].name  # TCxx
        base_id = mp4.stem             # TCxx_yyy

        # Decide whether to blur faces
        blur_faces = tc_name not in SOCIAL_CUES_TCS

        video_only_out = mp4.with_name(f"{base_id}_blurred_video_only.mp4")
        final_out = mp4.with_name(f"{base_id}_blurred.mp4")

        if SKIP_IF_EXISTS and final_out.exists():
            skipped += 1
            continue

        try:
            # 1) EgoBlur
            cmd = build_egoblur_cmd(mp4, video_only_out, blur_faces=blur_faces)
            print(f"\n▶ {tc_name} | blur_faces={blur_faces} | {mp4.name}")
            run(cmd)

            # 2) Re-mux audio (optional)
            if REMUX_AUDIO:
                try:
                    remux_audio_from_original(mp4, video_only_out, final_out)
                except subprocess.CalledProcessError:
                    # If original MP4 has no audio stream, keep video-only output as the final
                    # (but still write a reasonable "final" file for consistency).
                    print("⚠️ No audio stream (or remux failed). Keeping video-only output.")
                    video_only_out.replace(final_out)

            ok += 1

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {mp4} | {e}")
            failed += 1

    print("\nSummary")
    print(f"  Processed: {ok}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")

if __name__ == "__main__":
    main()
