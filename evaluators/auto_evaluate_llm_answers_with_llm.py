# -*- coding: utf-8 -*-
"""
Evaluate LLM predictions against ground-truth answers using an LLM-as-judge.

Usage:
  python evaluators/auto_evaluate_llm_answers_with_llm.py 9sets_human_vs_auto_combined.csv
"""

import os
import re
import sys
import pandas as pd
from difflib import SequenceMatcher  # retained for compatibility

# ---------- helpers ----------
def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, and remove punctuation for fair comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def is_multiple_choice(answer: str) -> bool:
    """Return True if answer looks like A/B/C/D style."""
    return isinstance(answer, str) and re.fullmatch(r"[a-dA-D]", answer.strip()) is not None

# ---------- LLM-backed judges (enhanced implementations) ----------
def _get_openai_client():
    """
    Lazy-create and memoize an OpenAI client using env var OPENAI_API_KEY.
    Returns None if key/SDK is missing; callers handle fallback.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # requires openai>=1.0
    except Exception:
        return None
    if getattr(_get_openai_client, "_client", None) is None:
        _get_openai_client._client = OpenAI(api_key=api_key)
    return _get_openai_client._client

def match_multiple_choice(prediction: str, answer: str) -> bool:
    """
    Judge MCQ correctness. Fast-path simple matches; otherwise ask a lightweight LLM to return 0/1.
    """
    if not isinstance(prediction, str) or not isinstance(answer, str):
        return False
    pred_s, ans_s = prediction.strip(), answer.strip()

    # fast path: normalized equality (e.g., "A" vs "a")
    if normalize_text(pred_s) == normalize_text(ans_s):
        return True

    # fast path: letter-vs-letter like A, A., (A), A)
    letter_pat = re.compile(r"^[\(\s]*([A-Da-d])[\)\.\s]*$")
    mg, mp = letter_pat.match(ans_s), letter_pat.match(pred_s)
    if mg and mp and mg.group(1).lower() == mp.group(1).lower():
        return True

    # LLM fallback (strict 0/1)
    client = _get_openai_client()
    if client is None:
        return False

    system = "You are a strict evaluator. Reply with 1 if correct, 0 if incorrect. No explanations."
    user = (
        "You will be given the TRUE answer key (a multiple-choice letter or exact option) "
        "and a PREDICTED answer. Output 1 if predicted matches the true answer (letter or the same option text), "
        "else 0.\n"
        f"TRUE: {answer}\nPREDICTED: {prediction}\nOutput 0 or 1 only:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=4,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out.startswith("1")
    except Exception:
        return False

def semantic_match(prediction: str, answer: str, threshold: float = 0.85) -> bool:
    """
    Judge open-ended correctness. Fast-path equality and ±1% numeric tolerance; otherwise ask an LLM for 0/1.
    """
    if not isinstance(prediction, str) or not isinstance(answer: str):
        return False
    p_norm, a_norm = normalize_text(prediction), normalize_text(answer)
    if not p_norm or not a_norm:
        return False
    if p_norm == a_norm:
        return True

    # numeric tolerance: treat numbers within ±1% as equal if units match or absent
    num_re = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z%]*)\s*$")
    def _parse_num(s):
        m = num_re.match((s or "").strip())
        if not m:
            return None
        return float(m.group(1)), (m.group(2) or "").lower()

    pa, pb = _parse_num(answer), _parse_num(prediction)
    if pa and pb:
        (va, ua), (vb, ub) = pa, pb
        if not (ua and ub and ua != ub):
            tol = abs(va) * 0.01  # 1%
            if (va == 0 and vb == 0) or abs(vb - va) <= tol:
                return True

    client = _get_openai_client()
    if client is None:
        return False

    system = "You are a strict evaluator. Reply with 1 if correct, 0 if incorrect. No explanations."
    user = (
        "You will be given TRUE and PREDICTED answers. "
        "Return 1 if PREDICTED is factually equivalent to TRUE (allow paraphrases; consider units; "
        "treat close numeric values as equivalent), else 0.\n"
        f"TRUE: {answer}\nPREDICTED: {prediction}\nOutput 0 or 1 only:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=4,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out.startswith("1")
    except Exception:
        return False

# ---------- evaluator (I/O and reporting flow preserved) ----------
def auto_evaluate(csv_path: str, dataset_path: str = None, output_path: str = None, verbose: bool = True):
    df = pd.read_csv(csv_path, encoding="latin-1")

    if "prediction" not in df.columns:
        raise ValueError("CSV must contain column named 'prediction'")
    
    if "question" not in df.columns or "answer" not in df.columns:
        if verbose: print("'question' or 'answer' column not found in CSV. Trying to add from the original dataset...")
        if dataset_path is None:
            dataset_name = os.path.basename(csv_path).split("_pred")[0]
            dataset_path = f"./data/benchmarks_data/{dataset_name}.csv"
            if os.path.exists(dataset_path):
                dfd = pd.read_csv(dataset_path)
                if verbose: print(f"Loaded original dataset from {dataset_path}")
            else:
                if verbose: print(f"Original dataset not found at {dataset_path}. Proceeding without questions.")
        else:
            dfd = pd.read_csv(dataset_path)
            if verbose: print(f"Loaded original dataset from {dataset_path}")
        if 'id' in df.columns and 'id' in dfd.columns:
            if 'question' not in df.columns:
                merge_columns = ['id', 'question']
                df = dfd[merge_columns].merge(df, on='id', how='left')
                if verbose: print("Merged 'question' column into predictions DataFrame.")
            if 'answer' not in df.columns:
                merge_columns = ['id', 'answer']
                df = dfd[merge_columns].merge(df, on='id', how='left')
                if verbose: print("Merged 'answer' column into predictions DataFrame.")
            if 'answer_type' not in df.columns and 'answer_type' in dfd.columns:
                merge_columns = ['id', 'answer_type']
                df = dfd[merge_columns].merge(df, on='id', how='left')
                if verbose: print("Merged 'answer_type' column into predictions DataFrame.")

    results = []
    df = df.dropna(subset=["prediction", "answer"])
    for _, row in df.iterrows():
        ans, pred = str(row["answer"]), str(row["prediction"])
        
        # is this a multiple-choice question?
        try:
            this_is_multiple_choice = row.get("answer_type", None) in ["multiple_choice", "multipleChoice"]
        except:
            this_is_multiple_choice = is_multiple_choice(ans)

        # decision via enhanced judges (signatures preserved)
        if this_is_multiple_choice:
            correct = 1 if match_multiple_choice(pred, ans) else 0
        else:
            correct = 1 if semantic_match(pred, ans) else 0

        results.append(correct)

    df["Auto Correct/Incorrect"] = results
    
    if "Correct/Incorrect" in df.columns:
        df["Different Auto vs Human"] = (df["Auto Correct/Incorrect"] != df["Correct/Incorrect"]).astype(int)
        if df["Different Auto vs Human"].sum() > 0:
            print("Discrepancies between human and auto evaluation:")
            print(df[df["Different Auto vs Human"] == 1][["answer", "prediction", "Correct/Incorrect", "Auto Correct/Incorrect"]])
        else:
            print("No discrepancies between human and auto evaluation.")

    # Save output
    if output_path is None:
        output_path = csv_path.replace(".csv", "_auto_eval.csv")

    df.to_csv(output_path, index=False)
    if verbose: print(f"Evaluation complete. Saved to: {output_path}")
    acc_auto = sum(results)/len(df)*100
    if verbose: print(f"Total questions: {len(df)} | Auto-found Correct: {sum(results)} | Accuracy: {acc_auto:.1f}%")
    if "Correct/Incorrect" in df.columns:
        if verbose: print(f"Total questions: {len(df)} | Human-found Correct: {int(sum(df['Correct/Incorrect']))} | Accuracy: {sum(df['Correct/Incorrect'])/len(df)*100:.1f}%")
        if verbose: print(f"Total discrepancies with human evaluation: {df['Different Auto vs Human'].sum()}, or {df['Different Auto vs Human'].sum()/len(df)*100:.1f}%")

    return acc_auto

if __name__ == "__main__":
    input_csv = sys.argv[1]
    acc_auto = auto_evaluate(input_csv, dataset_path=sys.argv[2] if len(sys.argv) > 2 else None, verbose=True)
    print(f"Auto-evaluation accuracy: {acc_auto:.1f}%")
