"""

Purpose: Automatically evaluate LLM predictions against ground truth answers.
Adds a new column 'Auto Correct/Incorrect' to the CSV:
  - 1 if the LLM prediction is correct
  - 0 if incorrect

Handles both multiple-choice and free-text questions.
"""

import pandas as pd
import re
from difflib import SequenceMatcher
import sys
import os


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


def match_multiple_choice(prediction: str, answer: str) -> bool:
    """Check if the correct choice letter (A/B/C/D) appears at the start of prediction."""
    if not isinstance(prediction, str):
        return False
    prediction = prediction.strip()
    # Look for pattern like "A", "A.", "A)", or "Answer: A"
    return re.match(rf"^\s*{answer}\b|^\s*{answer}[.)]", prediction.strip(), re.IGNORECASE) is not None


def semantic_match(prediction: str, answer: str, threshold: float = 0.85) -> bool:
    """
    For open-ended questions, use fuzzy string similarity to judge correctness.
    Threshold can be tuned; 0.85 works well for short factual answers.
    """
    p, a = normalize_text(prediction), normalize_text(answer)
    if not p or not a:
        return False
    ratio = SequenceMatcher(None, p, a).ratio()
    return ratio >= threshold


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
