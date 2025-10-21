"""
DeepSeek-R1-Distill-Qwen-7B benchmark model for HLE Bio/Med dataset.
Compatible with: python main.py --predict --models deepseek_r1_distill_qwen_7b

Author: Casual Dineshbhai Kalotra
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------
# File paths (absolute HPC-safe paths)
# ---------------------------------------------------------------------
BASE_DIR = "/courses/EAI6080.202615/students/kalotra.c/benchmark_project/Repositories/EAI6080_Fall2025"
DATA_PATH = os.path.join(BASE_DIR, "data/benchmarks_data/HLE_bio_med.csv")
OUT_DIR = os.path.join(BASE_DIR, "data/predictions_and_evaluations")

os.makedirs(OUT_DIR, exist_ok=True)
PRED_PATH = os.path.join(OUT_DIR, "HLE_bio_med_pred_deepseek_r1_distill_qwen_7b.csv")
METRICS_PATH = os.path.join(OUT_DIR, "HLE_bio_med_metrics_deepseek_r1_distill_qwen_7b.json")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def build_prompt(question: str) -> str:
    """Return a concise biomedical reasoning prompt."""
    return (
        "You are a biomedical reasoning assistant. "
        "Read the question carefully and choose the best option.\n\n"
        f"Question:\n{question}\n\n"
        "Answer with only the letter: A, B, C, or D."
    )


def detect_label_column(df: pd.DataFrame) -> str:
    """Detect which column contains the gold (correct) answer."""
    for c in ["ground_truth_answer", "label", "answer"]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Dataset missing gold answer column. Expected one of "
        f"['ground_truth_answer', 'label', 'answer'], but found: {list(df.columns)}"
    )


def extract_letter(text: str) -> str:
    """Extract the first capital letter (A/B/C/D) from model output."""
    for ch in "ABCD":
        if ch in text:
            return ch
    for ch in "abcd":
        if ch in text:
            return ch.upper()
    return "?"


# ---------------------------------------------------------------------
# Main predict() entry point (called by main.py)
# ---------------------------------------------------------------------
def predict():
    """Run DeepSeek-R1-Distill-Qwen-7B on HLE Bio/Med and save predictions."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {DATA_PATH} | {len(df)} rows")

    label_col = detect_label_column(df)
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    print(f"🔄 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    preds = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
        q = str(row["question"])
        gold = str(row[label_col]).strip().upper()
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_letter(raw)
        preds.append({
            "index": i,
            "question": q,
            "gold": gold,
            "prediction": pred,
            "correct": int(pred == gold),
            "raw_output": raw.strip(),
        })

    df_out = pd.DataFrame(preds)
    df_out.to_csv(PRED_PATH, index=False)
    acc = df_out["correct"].mean() if len(df_out) > 0 else 0.0
    metrics = {"accuracy": float(acc), "total": len(df_out)}

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📁 Saved predictions → {PRED_PATH}")
    print(f"📊 Accuracy: {acc:.3f} ({metrics['total']} samples)")
    print(f"📄 Metrics saved → {METRICS_PATH}")

    return metrics


# ---------------------------------------------------------------------
# Safety: prevent code from executing on import
# ---------------------------------------------------------------------
if __name__ == "__main__":
    predict()
