import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
import argparse
import base64
import io
import time
from PIL import Image
from tqdm import tqdm
from collections import Counter
from bert_score import score as bert_score_batch

############################################
# CONFIG
############################################

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llava")
args = parser.parse_args()

MODEL_TYPE = args.model.lower()

DATASET_ROOT = "vista_dataset"
OUTPUT_FILE = f"vista_results_{MODEL_TYPE}.csv"
ERROR_FILE = OUTPUT_FILE.replace(".csv", "_errors.csv")

NUM_FRAMES = 8
KEYFRAMES = 3
SEED = 42

GPT4O_MAX_RETRIES = 3
GPT4O_RETRY_DELAY = 5  # seconds between retries (doubles each attempt)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

############################################
# MODEL LOADING
############################################

if MODEL_TYPE == "llava":

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )

    model.eval()

elif MODEL_TYPE == "blip":

    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

    processor = Blip2Processor.from_pretrained(MODEL_NAME)

    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

elif MODEL_TYPE == "qwen":

    from transformers import AutoProcessor, AutoModelForVision2Seq

    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

elif MODEL_TYPE == "gpt4o":

    from openai import OpenAI, RateLimitError, APIError
    client = OpenAI()

else:
    raise ValueError(
        f"Unknown model type: '{MODEL_TYPE}'. Choose from: llava, blip, qwen, gpt4o"
    )

############################################
# FRAME EXTRACTION
############################################

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    return frames


def select_keyframes(frames):
    """Return KEYFRAMES evenly-spaced frames from the full frame list."""

    if len(frames) <= KEYFRAMES:
        return frames

    idx = np.linspace(0, len(frames) - 1, KEYFRAMES, dtype=int)

    return [frames[i] for i in idx]

############################################
# TOKEN F1
############################################

def token_f1(pred, gt):

    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)

    common = sum((pred_counts & gt_counts).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)

############################################
# PROMPTS — NO GROUND TRUTH LEAKAGE
############################################

def build_scene_prompt():
    """
    Blind scene prompt. No scene_description passed in — the model
    must describe the environment from visual input alone.
    """
    return (
        "You are assisting a visually impaired person. "
        "Describe the key objects, obstacles, layout, and any hazards "
        "visible in this environment based solely on what you observe."
    )


def build_guidance_prompt():
    """
    Blind guidance prompt. No action instructions passed in — the model
    must infer the task and provide guidance from visual input alone.
    """
    return (
        "You are assisting a visually impaired person. "
        "Based on what you observe in these images, provide clear "
        "step-by-step guidance to help them safely navigate or interact "
        "with this environment. Highlight any hazards or important landmarks."
    )

############################################
# GPT-4O WITH EXPONENTIAL BACKOFF RETRY
############################################

def call_gpt4o(img_b64, prompt):
    """Call GPT-4o with retry on rate limit or transient API errors."""

    from openai import RateLimitError, APIError

    for attempt in range(GPT4O_MAX_RETRIES):

        try:

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=256
            )

            return response.choices[0].message.content

        except (RateLimitError, APIError) as e:

            if attempt < GPT4O_MAX_RETRIES - 1:
                wait = GPT4O_RETRY_DELAY * (2 ** attempt)
                print(f"[GPT-4o] {type(e).__name__} — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{GPT4O_MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

############################################
# MODEL ROUTER — MULTI-FRAME
############################################

def run_model(frames, prompt):
    """
    Run inference using the selected model across multiple keyframes.
    Local models (LLaVA, Qwen) receive all KEYFRAMES via multiple <image>
    tokens with single-frame fallback if batched input fails.
    BLIP2 and GPT-4o are single-image by design.
    """

    keyframes = select_keyframes(frames)

    if MODEL_TYPE == "llava":

        image_tokens = "\n".join(["<image>"] * len(keyframes))
        full_prompt = f"USER: {image_tokens}\n{prompt}\nASSISTANT:"

        try:
            inputs = processor(
                text=full_prompt,
                images=keyframes,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

            text = processor.decode(out[0], skip_special_tokens=True)

        except Exception as e:
            print(f"[LLaVA] Multi-frame failed ({e}), falling back to single frame.")
            image = keyframes[len(keyframes) // 2]
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

            inputs = processor(
                text=full_prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

            text = processor.decode(out[0], skip_special_tokens=True)

        return text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()

    elif MODEL_TYPE == "blip":

        # BLIP2 is inherently single-image; use middle keyframe
        image = keyframes[len(keyframes) // 2]

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)

        return processor.decode(out[0], skip_special_tokens=True)

    elif MODEL_TYPE == "qwen":

        image_tokens = "".join(["<image>"] * len(keyframes))
        full_prompt = f"{image_tokens}\n{prompt}"

        try:
            inputs = processor(
                text=full_prompt,
                images=keyframes,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

            return processor.decode(out[0], skip_special_tokens=True)

        except Exception as e:
            print(f"[Qwen] Multi-frame failed ({e}), falling back to single frame.")
            image = keyframes[len(keyframes) // 2]
            full_prompt = f"<image>\n{prompt}"

            inputs = processor(
                text=full_prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

            return processor.decode(out[0], skip_special_tokens=True)

    elif MODEL_TYPE == "gpt4o":

        # Single middle frame for GPT-4o (API cost constraint)
        image = keyframes[len(keyframes) // 2]

        img_pil = Image.fromarray(image)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return call_gpt4o(img_b64, prompt)

############################################
# FIND CLIPS
############################################

def find_clips():

    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    clips = []

    for tc in sorted(os.listdir(DATASET_ROOT)):

        tc_path = os.path.join(DATASET_ROOT, tc)

        if not os.path.isdir(tc_path):
            continue

        for clip in sorted(os.listdir(tc_path)):

            clip_path = os.path.join(tc_path, clip)

            if not os.path.isdir(clip_path):
                continue

            video = None
            json_file = None

            for f in os.listdir(clip_path):

                if f.endswith(".mp4"):
                    video = os.path.join(clip_path, f)

                if f.endswith(".json") and not f.startswith("._"):
                    json_file = os.path.join(clip_path, f)

            if video and json_file:
                clips.append({
                    "clip_id": clip,
                    "video":   video,
                    "json":    json_file
                })

    return clips

############################################
# MAIN EVALUATION
############################################

def evaluate():

    clips = find_clips()
    print(f"Model             : {MODEL_TYPE}")
    print(f"Total clips found : {len(clips)}")

    # Build prompts once — identical blind prompt used for every clip
    scene_prompt    = build_scene_prompt()
    guidance_prompt = build_guidance_prompt()

    results = []
    errors  = []
    skipped = 0

    for clip in tqdm(clips):

        try:

            with open(clip["json"]) as f:
                data = json.load(f)

            if not data.get("annotations"):
                skipped += 1
                continue

            ann = data["annotations"][0].get("video_level", {})

            if not ann or not ann.get("question") or not ann.get("answer"):
                skipped += 1
                continue

            question    = ann.get("question", "")
            answer      = ann.get("answer", "")
            scene_gt    = ann.get("scene_description", "")
            actions     = ann.get("actions", [])
            guidance_gt = " ".join(a["instruction"] for a in actions)

            frames = extract_frames(clip["video"])

            scene_pred    = run_model(frames, scene_prompt)
            qa_pred       = run_model(frames, question)
            guidance_pred = run_model(frames, guidance_prompt)

            results.append({
                "clip_id":       clip["clip_id"],
                "scene_gt":      scene_gt,
                "scene_pred":    scene_pred,
                "qa_gt":         answer,
                "qa_pred":       qa_pred,
                "qa_exact":      int(qa_pred.strip().lower() == answer.strip().lower()),
                "qa_f1":         token_f1(qa_pred, answer),
                "guidance_gt":   guidance_gt,
                "guidance_pred": guidance_pred,
            })

        except Exception as e:

            errors.append({"clip_id": clip["clip_id"], "error": str(e)})
            print(f"[ERROR] {clip['clip_id']}: {e}")

    ############################################
    # EMPTY GUARD
    ############################################

    if not results:
        print("No results generated. Check dataset path and annotations.")
        if errors:
            pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)
            print(f"Saved {len(errors)} errors to {ERROR_FILE}")
        return

    df = pd.DataFrame(results)

    ############################################
    # BERTSCORE — .tolist() for clean serialization
    ############################################

    print("Computing BERTScore...")

    _, _, scene_f1 = bert_score_batch(
        df["scene_pred"].tolist(),
        df["scene_gt"].tolist(),
        lang="en",
        device=DEVICE
    )

    _, _, guidance_f1 = bert_score_batch(
        df["guidance_pred"].tolist(),
        df["guidance_gt"].tolist(),
        lang="en",
        device=DEVICE
    )

    df["scene_bert"]    = scene_f1.tolist()
    df["guidance_bert"] = guidance_f1.tolist()

    ############################################
    # SAVE
    ############################################

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

    if errors:
        pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)
        print(f"Errors saved to {ERROR_FILE}")

    ############################################
    # SUMMARY
    ############################################

    print("\n========== Benchmark Results ==========")
    print(f"Model                  : {MODEL_TYPE}")
    print("----------------------------------------")
    print(f"Total clips evaluated  : {len(df)}")
    print(f"Skipped (no annotation): {skipped}")
    print(f"Errors                 : {len(errors)}")
    print("----------------------------------------")
    print(f"Scene BERTScore        : {df['scene_bert'].mean():.4f}")
    print(f"QA Exact Match         : {df['qa_exact'].mean():.4f}")
    print(f"QA Token F1            : {df['qa_f1'].mean():.4f}")
    print(f"Guidance BERTScore     : {df['guidance_bert'].mean():.4f}")
    print("========================================")

############################################

if __name__ == "__main__":
    evaluate()