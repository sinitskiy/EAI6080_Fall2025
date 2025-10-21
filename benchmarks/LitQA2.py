"""
LitQA2 Benchmark Preparation Script
-----------------------------------
Prepares the LitQA2 benchmark dataset in CSV format for downstream LLM evaluation.
Follows the format:
id, question, image_path, answer, answer_type, category, raw_subject
"""

import os
import json
import pandas as pd


def prepare_LitQA2(dataset_path: str, output_dir: str = "data/benchmarks_data"):
    """
    Converts the LitQA2 dataset (JSONL/JSON/CSV) into a standardized CSV format.

    Parameters:
    -----------
    dataset_path : str
        Path to the original LitQA2 dataset file.
    output_dir : str
        Directory where the prepared CSV and any images (if present) will be saved.

    Output:
    -------
    A CSV file: data/benchmarks_data/LitQA2.csv
    """

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "LitQA2.csv")

    # --- Load dataset ---
    if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(records)
    elif dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("Unsupported dataset format. Use JSONL, JSON, or CSV.")

    # --- Required Columns ---
    id_list = []
    question_list = []
    answer_list = []
    image_path_list = []
    answer_type_list = []
    category_list = []
    raw_subject_list = []

    for i, row in df.iterrows():
        q = row.get("question", None)
        a = row.get("answer", None)
        img = row.get("image", None)
        a_type = row.get("answer_type", "multipleChoice" if "options" in row else "exactMatch")
        cat = row.get("category", row.get("subset", None))
        raw = row.get("raw_subject", None)

        id_list.append(i)
        question_list.append(q)
        answer_list.append(a)
        image_path_list.append(img if img else "")
        answer_type_list.append(a_type)
        category_list.append(cat)
        raw_subject_list.append(raw)

    # --- Construct final dataframe ---
    df_final = pd.DataFrame({
        "id": id_list,
        "question": question_list,
        "image_path": image_path_list,
        "answer": answer_list,
        "answer_type": answer_type_list,
        "category": category_list,
        "raw_subject": raw_subject_list
    })

    # --- Save output ---
    df_final.to_csv(output_csv, index=False)
    print(f"[✅] LitQA2 benchmark prepared successfully: {output_csv}")
    return df_final


# --- Script entry point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare LitQA2 benchmark for model evaluation.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the original LitQA2 dataset (JSON/JSONL/CSV).")
    parser.add_argument("--output_dir", type=str, default="data/benchmarks_data",
                        help="Output directory for prepared CSV.")
    args = parser.parse_args()

    prepare_LitQA2(args.dataset_path, args.output_dir)
