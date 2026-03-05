import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
import time
from collections import Counter
from tqdm import tqdm
from bert_score import score as bert_score_batch
from transformers import AutoProcessor, LlavaForConditionalGeneration

############################################
# CONFIGURATION
############################################

DATASET_ROOT = "vista_dataset"
OUTPUT_FILE = "vista_results.csv"
ERROR_FILE = OUTPUT_FILE.replace(".csv", "_errors.csv")

NUM_FRAMES = 8
SEED = 42

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

MAX_RETRIES = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# REPRODUCIBILITY
############################################

torch.manual_seed(SEED)
np.random.seed(SEED)

############################################
# LOAD MODEL
############################################

print("Loading VLM model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

############################################
# FRAME EXTRACTION
############################################

def extract_frames(video_path, num_frames=NUM_FRAMES):

    cap = cv2.VideoCapture(video_path)

    try:

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total == 0:
            raise ValueError(f"Video has 0 frames: {video_path}")

        indices = np.linspace(
            0,
            total - 1,
            num=min(num_frames, total),
            dtype=int
        )

        frames = []

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        if not frames:
            raise ValueError(f"No frames read from: {video_path}")

        return frames

    finally:
        cap.release()

############################################
# CHECK JSON VALIDITY
############################################

def json_has_annotation(json_path):

    try:

        with open(json_path) as f:
            data = json.load(f)

        if not data:
            return False

        if "annotations" not in data:
            return False

        if len(data["annotations"]) == 0:
            return False

        if "video_level" not in data["annotations"][0]:
            return False

        ann = data["annotations"][0]["video_level"]

        if not ann.get("question") or not ann.get("answer"):
            return False

        return True

    except Exception:  # FIX: bare except replaced with except Exception
        return False

############################################
# LOAD ANNOTATION
############################################

def load_annotation(json_path):

    with open(json_path) as f:
        data = json.load(f)

    ann = data["annotations"][0]["video_level"]

    return ann["question"], ann["answer"]

############################################
# TOKEN F1 METRIC
############################################

def token_f1(pred, gt):

    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()

    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)

    # Intersection takes minimum count per token, preserving duplicates
    common = sum((pred_counts & gt_counts).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)

############################################
# RUN VLM
############################################

def run_vlm(frames, question):
    """
    Use the middle frame as the representative frame for LLaVA-1.5,
    which only supports single-image input.
    """

    representative_frame = frames[len(frames) // 2]

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=representative_frame,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    full_text = processor.decode(output[0], skip_special_tokens=True)

    if "ASSISTANT:" in full_text:
        response = full_text.split("ASSISTANT:")[-1].strip()
    else:
        response = full_text.strip()

    return response

############################################
# FIND ALL CLIPS
############################################

def find_all_clips(dataset_root):

    # FIX: restore missing existence check from v1
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    clips = []

    for tc_folder in sorted(os.listdir(dataset_root)):

        tc_path = os.path.join(dataset_root, tc_folder)

        if not os.path.isdir(tc_path):
            continue

        for clip_folder in sorted(os.listdir(tc_path)):

            clip_path = os.path.join(tc_path, clip_folder)

            if not os.path.isdir(clip_path):
                continue

            video_file = None
            json_file = None

            for file in os.listdir(clip_path):

                if file.endswith(".mp4"):
                    video_file = os.path.join(clip_path, file)

                if file.endswith(".json"):
                    json_file = os.path.join(clip_path, file)

            if video_file and json_file:

                clips.append({
                    "clip_id": clip_folder,
                    "video": video_file,
                    "json": json_file
                })

            else:

                missing = []
                if not video_file:
                    missing.append("video (.mp4)")
                if not json_file:
                    missing.append("annotation (.json)")
                print(f"[WARN] Skipping {clip_folder}: missing {', '.join(missing)}")

    return clips

############################################
# RESUME: LOAD ALREADY-PROCESSED CLIP IDS
############################################

def load_processed_ids(output_file):
    """Return set of clip_ids already written to the output CSV."""

    if not os.path.exists(output_file):
        return set()

    try:
        df = pd.read_csv(output_file)
        return set(df["clip_id"].astype(str).tolist())
    except Exception:
        return set()

############################################
# MAIN EVALUATION
############################################

def evaluate():

    clips = find_all_clips(DATASET_ROOT)

    print(f"Total clips found: {len(clips)}")

    # FIX: checkpoint/final-save conflict resolved by:
    #   - appending per-clip results to CSV as we go (crash-safe)
    #   - reading back the full CSV at the end for BERTScore + summary
    #   - BERTScore written to a separate summary file, not overwriting the main CSV
    #   - supporting resume: skip clips already in the output file

    processed_ids = load_processed_ids(OUTPUT_FILE)

    if processed_ids:
        print(f"Resuming: {len(processed_ids)} clips already processed, skipping.")

    skipped_empty = 0
    errors = []
    newly_processed = []

    write_header = not os.path.exists(OUTPUT_FILE)

    for clip in tqdm(clips):

        # Skip already-processed clips (resume support)
        if clip["clip_id"] in processed_ids:
            continue

        try:

            if not json_has_annotation(clip["json"]):
                skipped_empty += 1
                continue

            question, gt = load_annotation(clip["json"])

            frames = extract_frames(clip["video"])

            ####################################
            # RETRY LOGIC
            ####################################

            pred = None
            latency = None

            for attempt in range(MAX_RETRIES):

                try:

                    start = time.time()
                    pred = run_vlm(frames, question)
                    latency = time.time() - start
                    break

                except RuntimeError as e:

                    print(f"[Retry {attempt+1}] {clip['clip_id']} : {e}")

                    if "CUDA" in str(e):
                        torch.cuda.empty_cache()

                    if attempt == MAX_RETRIES - 1:
                        raise

            # FIX: guard against pred/latency being None if retries somehow
            # exhaust without raising (defensive; raise above should prevent this)
            if pred is None or latency is None:
                raise RuntimeError("Inference did not produce a result after retries.")

            ####################################
            # METRICS
            ####################################

            exact_match = int(pred.strip().lower() == gt.strip().lower())

            f1 = token_f1(pred, gt)

            result = {
                "clip_id": clip["clip_id"],
                "question": question,
                "ground_truth": gt,
                "prediction": pred,
                "exact_match": exact_match,
                "token_f1": round(f1, 4),
                "latency_sec": round(latency, 3),
            }

            ####################################
            # APPEND TO CSV (crash-safe checkpoint)
            ####################################

            pd.DataFrame([result]).to_csv(
                OUTPUT_FILE,
                mode="a",
                header=write_header,
                index=False,
            )
            write_header = False  # only write header once

            newly_processed.append(result)

        except Exception as e:

            errors.append({
                "clip_id": clip["clip_id"],
                "error": str(e)
            })

            print(f"[ERROR] {clip['clip_id']}: {e}")

    ############################################
    # READ BACK FULL CSV FOR BERTSCORE + SUMMARY
    ############################################

    if not os.path.exists(OUTPUT_FILE):
        print("No results generated.")
        return

    df = pd.read_csv(OUTPUT_FILE)

    if df.empty:
        print("No results generated.")
        return

    print(f"\nTotal results in file: {len(df)}")

    print("Computing BERTScore...")

    predictions = df["prediction"].tolist()
    ground_truths = df["ground_truth"].tolist()

    _, _, F1 = bert_score_batch(
        predictions,
        ground_truths,
        lang="en",
        device=DEVICE,   # FIX: was hardcoded "cuda", now respects CPU fallback
        verbose=False,
    )

    df["bert_score_f1"] = [round(f, 4) for f in F1.tolist()]

    # FIX: write enriched results (with BERTScore) back to main CSV cleanly
    df.to_csv(OUTPUT_FILE, index=False)

    ############################################
    # PRINT SUMMARY
    ############################################

    print(f"Saved {len(df)} results to {OUTPUT_FILE}")
    print(f"Mean BERTScore F1 : {df['bert_score_f1'].mean():.4f}")
    print(f"Exact Match       : {df['exact_match'].mean():.4f}")
    print(f"Token F1          : {df['token_f1'].mean():.4f}")
    print(f"Average latency   : {df['latency_sec'].mean():.3f} sec")
    print(f"Skipped (no ann.) : {skipped_empty}")
    print(f"Errors this run   : {len(errors)}")

    ############################################
    # SAVE ERRORS
    ############################################

    if errors:

        # Append errors so previous runs aren't lost
        error_df = pd.DataFrame(errors)
        error_df.to_csv(
            ERROR_FILE,
            mode="a",
            header=not os.path.exists(ERROR_FILE),
            index=False,
        )

        print(f"Saved {len(errors)} errors to {ERROR_FILE}")

############################################
# RUN
############################################

if __name__ == "__main__":

    evaluate()