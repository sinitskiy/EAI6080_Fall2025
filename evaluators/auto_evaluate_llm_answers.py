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

# -------------------------------
# Helper functions
# -------------------------------

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


# -------------------------------
# Main logic
# -------------------------------

def auto_evaluate(csv_path: str, output_path: str = None):
    df = pd.read_csv(csv_path)

    if "prediction" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain columns named 'prediction' and 'answer'")

    results = []
    for _, row in df.iterrows():
        ans, pred = str(row["answer"]), str(row["prediction"])

        if is_multiple_choice(ans):
            correct = 1 if match_multiple_choice(pred, ans) else 0
        else:
            correct = 1 if semantic_match(pred, ans) else 0

        results.append(correct)

    df["Auto Correct/Incorrect"] = results

    # Save output
    if output_path is None:
        output_path = csv_path.replace(".csv", "_auto_eval.csv")

    df.to_csv(output_path, index=False)
    print(f"✅ Evaluation complete. Saved to: {output_path}")
    print(f"Total questions: {len(df)} | Correct: {sum(results)} | Accuracy: {sum(results)/len(df)*100:.1f}%")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Change this path to your input CSV
    input_csv = r"D:\ragqa-arena\rag-qa-arena\DeepSeekR1_RAGQA_Science.csv"
    auto_evaluate(input_csv)
