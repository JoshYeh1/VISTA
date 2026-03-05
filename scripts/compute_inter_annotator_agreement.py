import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

import torch
from bert_score import score as bert_score_batch
from sentence_transformers import SentenceTransformer

############################################
# CONFIGURATION
############################################

ANNOTATION_DIR = "/Volumes/T7 Shield/json_stuff/annotator_agreement"
OUTPUT_DIR     = "/Volumes/T7 Shield"

ANSWER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vista_answer_inter_annotator_agreement.csv")
SCENE_OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "vista_scene_inter_annotator_agreement.csv")

ANSWER_HISTOGRAM = os.path.join(OUTPUT_DIR, "answer_agreement_histogram.png")
SCENE_HISTOGRAM  = os.path.join(OUTPUT_DIR, "scene_agreement_histogram.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# LOAD SBERT MODEL
############################################

print("Loading SBERT model...")
sbert_model = SentenceTransformer("all-mpnet-base-v2")

############################################
# LOAD ANNOTATION TEXT
############################################

def load_annotation(json_path):

    try:
        with open(json_path) as f:
            data = json.load(f)

        annotations = data.get("annotations")

        if not annotations:
            return None, None

        ann = annotations[0]

        if ann.get("was_cancelled"):
            return None, None

        results = ann.get("result", [])

        answer = None
        scene  = None

        for r in results:

            field = r.get("from_name")
            text  = r.get("value", {}).get("text", [])

            if not text:
                continue

            text = text[0].strip()

            if field == "answer_whole":
                answer = text

            if field == "scene_description_whole":
                scene = text

        return answer, scene

    except Exception as e:
        print(f"[JSON ERROR] {json_path}: {e}")
        return None, None


############################################
# GROUP ANNOTATIONS BY CLIP
############################################

CLIP_ID_PATTERN = re.compile(r'^([A-Za-z0-9]+_[A-Za-z0-9]+)')

def group_annotations():

    groups = defaultdict(list)
    skipped = 0

    for file in os.listdir(ANNOTATION_DIR):

        if not file.endswith(".json"):
            continue

        match = CLIP_ID_PATTERN.match(file)

        if not match:
            print(f"[WARN] Filename did not match clip ID pattern, skipping: {file}")
            skipped += 1
            continue

        clip_id = match.group(1)

        groups[clip_id].append(
            os.path.join(ANNOTATION_DIR, file)
        )

    if skipped:
        print(f"[WARN] Total files skipped due to naming mismatch: {skipped}")

    return groups


############################################
# SBERT SIMILARITY
############################################

def sbert_similarity_batch(texts_a, texts_b):

    all_texts = texts_a + texts_b

    embeddings = sbert_model.encode(
        all_texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True
    )

    A = embeddings[:len(texts_a)]
    B = embeddings[len(texts_a):]

    similarities = np.sum(A * B, axis=1)

    return similarities.tolist()


############################################
# AGREEMENT COMPUTATION
############################################

def compute_agreement(texts_a, texts_b, clip_ids, annotators_a, annotators_b, output_csv, histogram_file, title):

    if len(texts_a) == 0:
        print(f"[WARNING] No valid pairs found for {title}")
        return

    print("\nComputing BERTScore...")

    _, _, F1 = bert_score_batch(
        texts_a,
        texts_b,
        lang="en",
        device=DEVICE,
        verbose=False,
    )

    F1 = F1.cpu().numpy()

    print("\nComputing SBERT similarity...")

    sbert_scores = sbert_similarity_batch(texts_a, texts_b)

    rows = []

    for i in range(len(texts_a)):

        rows.append({
            "clip_id": clip_ids[i],
            "annotator_A": annotators_a[i],
            "annotator_B": annotators_b[i],
            "annotation_A": texts_a[i],
            "annotation_B": texts_b[i],
            "bert_score_f1": round(float(F1[i]), 4),
            "sbert_similarity": round(float(sbert_scores[i]), 4),
        })

    df = pd.DataFrame(rows)

    df.to_csv(output_csv, index=False)

    ############################################
    # HISTOGRAM
    ############################################

    mean_sim = df["sbert_similarity"].mean()

    fig, ax = plt.subplots(figsize=(10,6))

    ax.hist(df["sbert_similarity"], bins=20, color="steelblue", edgecolor="white")

    ax.axvline(mean_sim, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_sim:.3f}")

    ax.set_title(title)
    ax.set_xlabel("SBERT Similarity")
    ax.set_ylabel("Number of Annotation Pairs")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(histogram_file, dpi=150)
    plt.close(fig)

    ############################################
    # SUMMARY
    ############################################

    print(f"\nResults saved to: {output_csv}")
    print(f"Histogram saved to: {histogram_file}")

    print("\n--- INTER-ANNOTATOR AGREEMENT SUMMARY ---")

    print(f"Pairs evaluated       : {len(df)}")
    print(f"Mean BERTScore F1     : {df['bert_score_f1'].mean():.4f}")
    print(f"Std BERTScore F1      : {df['bert_score_f1'].std():.4f}")
    print(f"Mean SBERT similarity : {df['sbert_similarity'].mean():.4f}")
    print(f"Std SBERT similarity  : {df['sbert_similarity'].std():.4f}")


############################################
# MAIN
############################################

def evaluate():

    groups = group_annotations()

    print(f"\nTotal overlapping clips: {len(groups)}")

    answer_a = []
    answer_b = []
    answer_clip_ids = []
    answer_ann_a = []
    answer_ann_b = []

    scene_a = []
    scene_b = []
    scene_clip_ids = []
    scene_ann_a = []
    scene_ann_b = []

    for clip_id, files in groups.items():

        annotations = []

        for f in files:

            answer, scene = load_annotation(f)

            annotator = os.path.basename(f).split("_")[-1].replace(".json","")

            annotations.append((annotator, answer, scene))

        for (ann_a, ans_a, sc_a), (ann_b, ans_b, sc_b) in combinations(annotations, 2):

            if ans_a and ans_b:

                answer_a.append(ans_a)
                answer_b.append(ans_b)

                answer_clip_ids.append(clip_id)
                answer_ann_a.append(ann_a)
                answer_ann_b.append(ann_b)

            if sc_a and sc_b:

                scene_a.append(sc_a)
                scene_b.append(sc_b)

                scene_clip_ids.append(clip_id)
                scene_ann_a.append(ann_a)
                scene_ann_b.append(ann_b)

    ###################################
    # ANSWER AGREEMENT
    ###################################

    compute_agreement(
        answer_a,
        answer_b,
        answer_clip_ids,
        answer_ann_a,
        answer_ann_b,
        ANSWER_OUTPUT_FILE,
        ANSWER_HISTOGRAM,
        "Inter-Annotator Agreement (Answer)"
    )

    ###################################
    # SCENE AGREEMENT
    ###################################

    compute_agreement(
        scene_a,
        scene_b,
        scene_clip_ids,
        scene_ann_a,
        scene_ann_b,
        SCENE_OUTPUT_FILE,
        SCENE_HISTOGRAM,
        "Inter-Annotator Agreement (Scene Description)"
    )


############################################
# RUN
############################################

if __name__ == "__main__":
    evaluate()