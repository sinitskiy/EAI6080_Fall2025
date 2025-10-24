"""
auto_evaluate_llm_answers.py
-----------------------------------------
Purpose:
Automatically evaluate LLM predictions against ground truth answers.

Adds columns:
  - 'Auto Correct/Incorrect' → 1 if correct, 0 if incorrect
  - 'Different Auto vs Human' → 1 if auto disagrees with human (if human column exists)

Handles both:
  • Multiple-choice questions (A/B/C/D style)
  • Free-text factual answers (semantic fuzzy matching)

Usage:
    python auto_evaluate_llm_answers.py 9sets_human_vs_auto_combined.csv
-----------------------------------------
"""

import pandas as pd
import re
import sys
import os
from difflib import SequenceMatcher


# -----------------------------------------------------------
#  Text normalization
# -----------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, and remove punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------------------------------------
#  Identify question type
# -----------------------------------------------------------
def is_multiple_choice(answer: str) -> bool:
    """Return True if answer looks like A/B/C/D."""
    return isinstance(answer, str) and re.fullmatch(r"[a-dA-D]", answer.strip()) is not None


# -----------------------------------------------------------
#  Improved multiple-choice match
# -----------------------------------------------------------
def match_multiple_choice(prediction: str, answer: str) -> bool:
    """Check if the correct letter (A–D) clearly appears in prediction."""
    if not isinstance(prediction, str) or not isinstance(answer, str):
        return False

    prediction = prediction.strip()

    # ✅ Match correct forms like "Answer: A", "Option A", "A. text", "A) text"
    positive_pattern = rf"(\b(answer|option|choice)\s*[:\-]?\s*{answer}\b)|(^\s*{answer}[\.\)\s])"
    if re.search(positive_pattern, prediction, re.IGNORECASE):
        return True

    # ❌ Avoid "A short answer" or "An example"
    negative_pattern = rf"^{answer}[a-z]"  # letter followed by another letter
    if re.match(negative_pattern, prediction, re.IGNORECASE):
        return False

    # Strict fallback: must be A. / A) or exactly A
    return bool(re.match(rf"^\s*{answer}([\.\)]|\b)", prediction, re.IGNORECASE))


# -----------------------------------------------------------
#  Semantic comparison for open-ended answers
# -----------------------------------------------------------
def semantic_match(prediction: str, answer: str, threshold: float = 0.75) -> bool:
    """Check fuzzy semantic similarity for text answers."""
    p, a = normalize_text(prediction), normalize_text(answer)
    if not p or not a:
        return False

    # Exact or substring match shortcut
    if a in p or p in a:
        return True

    # Allow partial overlap for long factual strings like sequences
    if len(a) > 20 and a[:15] in p:
        return True

    # Fuzzy similarity
    ratio = SequenceMatcher(None, p, a).ratio()
    return ratio >= threshold


# -----------------------------------------------------------
#  Main evaluation logic
# -----------------------------------------------------------
def auto_evaluate(csv_path: str, dataset_path: str = None, output_path: str = None, verbose: bool = True):
    df = pd.read_csv(csv_path, encoding="latin-1")

    if "prediction" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'prediction' and 'answer' columns.")

    results = []

    for _, row in df.iterrows():
        ans = str(row["answer"]).strip()
        pred = str(row["prediction"]).strip()

        # Identify type (explicit or inferred)
        is_mc = row.get("answer_type", "").lower() in ["multiple_choice", "multiplechoice"] if "answer_type" in row else is_multiple_choice(ans)

        if is_mc:
            correct = 1 if match_multiple_choice(pred, ans) else 0
        else:
            correct = 1 if semantic_match(pred, ans) else 0

        results.append(correct)

    df["Auto Correct/Incorrect"] = results

    # -------------------------------------------------------
    # Compare with human annotations if present
    # -------------------------------------------------------
    if "Correct/Incorrect" in df.columns:
        df["Different Auto vs Human"] = (df["Auto Correct/Incorrect"] != df["Correct/Incorrect"]).astype(int)
        diff_count = df["Different Auto vs Human"].sum()

        if diff_count > 0:
            print(f"\n⚠ Found {diff_count} discrepancies between auto and human evaluation.\n")
            print(df[df["Different Auto vs Human"] == 1][["answer", "prediction", "Correct/Incorrect", "Auto Correct/Incorrect"]].head(15))
        else:
            print("\n✅ No discrepancies between auto and human evaluation.\n")

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    if output_path is None:
        output_path = csv_path.replace(".csv", "_auto_eval.csv")

    df.to_csv(output_path, index=False)
    if verbose:
        print(f"✅ Evaluation complete. Saved to: {output_path}")

    # -------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------
    total = len(df)
    correct_auto = sum(results)
    acc_auto = correct_auto / total * 100 if total else 0
    print(f"Total questions: {total} | Auto-found Correct: {correct_auto} | Accuracy: {acc_auto:.1f}%")

    if "Correct/Incorrect" in df.columns:
        human_acc = df["Correct/Incorrect"].sum() / total * 100
        print(f"Human Accuracy: {human_acc:.1f}% | Discrepancies: {diff_count} ({diff_count/total*100:.1f}%)")

    return acc_auto


# -----------------------------------------------------------
#  CLI Entry
# -----------------------------------------------------------
# -----------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Auto Evaluation Script (IDLE mode)\n")
    # 👉 Either hardcode your CSV path here, or type it interactively:
    # Example hardcode:
    csv_path = r"C:\Users\ratna\OneDrive\Desktop\Sonnet 4.5 Images\9sets_human_vs_auto_combined.csv"

    # Run the evaluator
    acc_auto = auto_evaluate(csv_path, verbose=True)
    print(f"\nAuto-evaluation accuracy: {acc_auto:.1f}%")
    print("\n✅ Done! Check your new file ending with _auto_eval.csv in the same folder.")
