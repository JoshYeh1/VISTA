import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from bert_score import score as bert_score_batch
from transformers import AutoProcessor, LlavaForConditionalGeneration

############################################
# CONFIGURATION
############################################

DATASET_ROOT = "vista_dataset"
OUTPUT_FILE = "vista_results.csv"

NUM_FRAMES = 8
SEED = 42

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

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
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

############################################
# FRAME EXTRACTION
############################################

def extract_frames(video_path, num_frames=NUM_FRAMES):
    """
    Extract evenly spaced frames from a video.
    Uses linspace so short videos still work.
    """

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
            raise ValueError(f"No frames could be read from: {video_path}")

        return frames

    finally:
        cap.release()

############################################
# LOAD ANNOTATION
############################################

def load_annotation(json_path):

    with open(json_path) as f:
        data = json.load(f)

    ann = data["annotations"][0]["video_level"]

    question = ann["question"]
    answer = ann["answer"]

    return question, answer


############################################
# RUN VLM
############################################

def run_vlm(frames, question):
    """
    Run VLM on a representative frame.
    LLaVA-1.5 supports a single image input.
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

    clips = []

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

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
# MAIN EVALUATION
############################################

def evaluate():

    clips = find_all_clips(DATASET_ROOT)

    print(f"Total clips found: {len(clips)}")

    results = []
    errors = []

    for clip in tqdm(clips):

        try:

            question, gt = load_annotation(clip["json"])

            frames = extract_frames(clip["video"])

            start = time.time()

            pred = run_vlm(frames, question)

            latency = time.time() - start

            results.append({
                "clip_id": clip["clip_id"],
                "question": question,
                "ground_truth": gt,
                "prediction": pred,
                "latency_sec": round(latency, 3)
            })

        except Exception as e:

            errors.append({
                "clip_id": clip["clip_id"],
                "error": str(e)
            })

            print(f"[ERROR] {clip['clip_id']}: {e}")

    if not results:
        print("No results generated.")
        return

    ############################################
    # BERTSCORE (BATCH)
    ############################################

    print("Computing BERTScore...")

    predictions = [r["prediction"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]

    _, _, F1 = bert_score_batch(
        predictions,
        ground_truths,
        lang="en",
        device="cuda",
        verbose=False
    )

    for r, f1 in zip(results, F1.tolist()):
        r["bert_score_f1"] = round(f1, 4)

    ############################################
    # SAVE RESULTS
    ############################################

    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df)} results to {OUTPUT_FILE}")

    print(f"Mean BERTScore F1: {df['bert_score_f1'].mean():.4f}")

    print(f"Average latency: {df['latency_sec'].mean():.3f} sec")

    ############################################
    # SAVE ERRORS IF ANY
    ############################################

    if errors:

        error_file = OUTPUT_FILE.replace(".csv", "_errors.csv")

        pd.DataFrame(errors).to_csv(error_file, index=False)

        print(f"Saved {len(errors)} errors to {error_file}")


############################################
# RUN
############################################

if __name__ == "__main__":

    evaluate()