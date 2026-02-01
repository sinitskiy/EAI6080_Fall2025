"""
auto_evaluate_llm_answers.py — Smart Hybrid (Final Tuned)
Author: Ratna Sekhar
"""

import pandas as pd
import re
from difflib import SequenceMatcher


# -----------------------------------------------------------
# Normalize text
# -----------------------------------------------------------
def normalize_text(text: str, keep_symbols=False) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    if keep_symbols:
        text = re.sub(r"[^a-z0-9=\+\-\^\(\)\[\]\./\\,]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------------
# Identify multiple-choice
# -----------------------------------------------------------
def is_multiple_choice(answer: str) -> bool:
    return isinstance(answer, str) and re.fullmatch(r"[a-jA-J]", answer.strip()) is not None


# -----------------------------------------------------------
# Refined multiple-choice matching
# -----------------------------------------------------------
def match_multiple_choice(prediction: str, answer: str) -> bool:
    if not isinstance(prediction, str) or not isinstance(answer, str):
        return False

    answer = answer.strip().upper()
    pred = prediction.strip()

    # Remove filler tokens
    pred = re.sub(r"(?i)(answer|option|choice|selected|is|correct)\s*[:\-]?\s*", "", pred).strip()

    # Reject multi-option outputs (A or B, A/B, etc.)
    if re.search(r"\b[A-J](\)|\.|,|\s)*(and|or|,|/)\s*[A-J]\b", pred, re.IGNORECASE):
        return False

    # Direct exact letter match
    if pred.strip().upper() == answer:
        return True

    # Accept typical LLM patterns like "A.", "A)", "(A)", "Option A"
    if re.match(rf"^\s*\(?{answer}[\.\)\s]", pred, re.IGNORECASE):
        return True
    if re.search(rf"\b(answer|option|choice)\s*[:\-]?\s*{answer}\b", pred, re.IGNORECASE):
        return True

    return False


# -----------------------------------------------------------
# Advanced semantic match (dynamic threshold)
# -----------------------------------------------------------
def semantic_match(prediction: str, answer: str) -> bool:
    if not isinstance(prediction, str) or not isinstance(answer, str):
        return False

    # keep symbols if equation, numeric, or short
    symbolic = bool(re.search(r"[\=\+\-\^\d]", answer)) or len(answer) <= 5
    p = normalize_text(prediction, keep_symbols=symbolic)
    a = normalize_text(answer, keep_symbols=symbolic)

    if not p or not a:
        return False

    # 1️⃣ Exact numeric or symbol-based → strict
    if a.isdigit():
        return p == a

    # 2️⃣ Chem/sequence/symbolic terms → relaxed substring
    if re.search(r"[a-z]\d|\d[a-z]|\-", a) and a in p:
        return True

    # 3️⃣ Exact or contained (for long factuals)
    if a == p or (len(a) > 10 and a in p):
        return True

    # 4️⃣ Short factual/numeric words (2–10 chars): high precision threshold
    ratio = SequenceMatcher(None, p, a).ratio()
    if len(a) <= 10:
        return ratio >= 0.93

    # 5️⃣ Longer answers: moderate fuzzy threshold
    if len(a) > 10:
        return ratio >= 0.86

    return False


# -----------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------
def auto_evaluate(csv_path: str, output_path: str = None, verbose: bool = True):
    df = pd.read_csv(csv_path, encoding="latin-1")

    if "prediction" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'prediction' and 'answer' columns.")

    results = []
    for _, row in df.iterrows():
        ans = str(row["answer"]).strip()
        pred = str(row["prediction"]).strip()
        is_mc = row.get("answer_type", "").lower() in ["multiple_choice", "multiplechoice"] if "answer_type" in row else is_multiple_choice(ans)

        correct = 1 if (match_multiple_choice(pred, ans) if is_mc else semantic_match(pred, ans)) else 0
        results.append(correct)

    df["Auto Correct/Incorrect"] = results

    diff_count = 0
    if "Correct/Incorrect" in df.columns:
        df["Different Auto vs Human"] = (df["Auto Correct/Incorrect"] != df["Correct/Incorrect"]).astype(int)
        diff_count = df["Different Auto vs Human"].sum()
        if diff_count > 0:
            print(f"\n⚠ Found {diff_count} discrepancies between auto and human evaluation.\n")
            print(df[df["Different Auto vs Human"] == 1][["answer", "prediction", "Correct/Incorrect", "Auto Correct/Incorrect"]].head(15))
        else:
            print("\n✅ No discrepancies between auto and human evaluation.\n")

    if output_path is None:
        output_path = csv_path.replace(".csv", "_auto_eval.csv")

    df.to_csv(output_path, index=False)
    if verbose:
        print(f"✅ Evaluation complete. Saved to: {output_path}")

    total = len(df)
    acc_auto = sum(results) / total * 100
    print(f"Total: {total} | Auto Correct: {sum(results)} | Accuracy: {acc_auto:.1f}%")
    if "Correct/Incorrect" in df.columns:
        human_acc = df["Correct/Incorrect"].sum() / total * 100
        print(f"Human Accuracy: {human_acc:.1f}% | Discrepancies: {diff_count} ({diff_count/total*100:.1f}%)")

    return acc_auto


# -----------------------------------------------------------
# IDLE Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Auto Evaluation Script (IDLE mode)\n")
    csv_path = r"C:\Users\ratna\OneDrive\Desktop\Sonnet 4.5 Images\9sets_human_vs_auto_combined.csv"
    acc_auto = auto_evaluate(csv_path, verbose=True)
    print(f"\nAuto-evaluation accuracy: {acc_auto:.1f}%")
    print("\n✅ Done! Check your new file ending with _auto_eval.csv in the same folder.")
