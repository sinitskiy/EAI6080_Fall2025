import os
import csv
from datasets import load_dataset

def prepare(output_path="data/benchmarks_data/cvqa.csv",
            image_dir="data/benchmarks_data/cvqa"):
    """
    Prepare the CVQA benchmark dataset and save it in the required CSV format.
    Columns: id, question, image_path, answer, answer_type, category, raw_subject
    """

    # Load the dataset
    dataset = load_dataset("afaji/cvqa", split="validation")

    # Make sure output folders exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "question", "image_path", "answer",
            "answer_type", "category", "raw_subject"
        ])

        for i, sample in enumerate(dataset):
            q_id = f"CVQA_{i+1}"
            question = sample.get("question", "")
            answer = sample.get("answer", "")

            # Image handling: if images exist in dataset
            img_path = ""
            if "image" in sample and sample["image"] is not None:
                img_filename = f"{q_id}.jpg"
                img_path = os.path.join(image_dir, img_filename)
                try:
                    sample["image"].save(img_path)
                except Exception:
                    img_path = ""

            # Defaults if fields not available
            answer_type = sample.get("answer_type", "exactMatch")
            category = sample.get("category", "val")
            raw_subject = sample.get("subset", "")

            writer.writerow([
                q_id, question, img_path, answer,
                answer_type, category, raw_subject
            ])

    print(f"✅ CVQA benchmark saved to {output_path}")
