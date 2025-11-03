import os
import csv
from datasets import load_dataset
from openai import OpenAI

def predict(output_path="data/predictions_and_evaluations/cvqa_pred_gpt_5_mini.csv"):
    """
    Run GPT-5-mini on the CVQA benchmark and save predictions.
    Columns: id, answer (ground truth), prediction
    """

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load CVQA dataset
    dataset = load_dataset("afaji/cvqa", split="validation")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer", "prediction"])

        # For demo: first 20 rows. For full dataset, remove .select(range(20))
        for i, sample in enumerate(dataset.select(range(20))):
            q_id = f"CVQA_{i+1}"
            question = sample.get("question", "")
            gt_answer = sample.get("answer", "")

            try:
                # Query GPT-5-mini
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": question}],
                    max_tokens=100
                )
                prediction = response.choices[0].message.content.strip()
            except Exception as e:
                prediction = f"ERROR: {str(e)}"

            writer.writerow([q_id, gt_answer, prediction])

    print(f"✅ Predictions saved to {output_path}")
