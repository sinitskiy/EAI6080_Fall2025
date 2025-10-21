import json
import csv
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # -----------------------------
    # Load dataset from Parquet folder
    # -----------------------------
    print(f"🔹 Loading HLE-Gold-Bio-Chem dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)

    # Merge train + test if both exist
    if "train" in dataset and "test" in dataset:
        data = dataset["train"] + dataset["test"]
    else:
        first_split = list(dataset.keys())[0]
        data = dataset[first_split]

    print(f"📘 Loaded {len(data)} questions total.")

    # -----------------------------
    # Load local model
    # -----------------------------
    print(f"🔗 Loading local model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    # -----------------------------
    # Generate predictions
    # -----------------------------
    output_rows = []
    for i, item in enumerate(data, 1):
        question = item.get("question", "")
        answer = item.get("answer", "")
        answer_type = item.get("answer_type", "")
        subject = item.get("subject", "")
        category = item.get("category", "")
        canary = item.get("canary", "")

        # Detect MCQ vs free form
        if "A." in question or "A)" in question or "A:" in question:
            prompt = f"Question:\n{question}\n\nChoose the correct answer (A, B, C, or D) and explain.\nAnswer:"
        else:
            prompt = f"Question:\n{question}\n\nAnswer the question and explain your reasoning.\nAnswer:"

        try:
            result = pipe(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )[0]["generated_text"]

            if "Answer:" in result:
                prediction = result.split("Answer:", 1)[-1].strip()
            else:
                prediction = result.strip()

            prediction = f"Answer: {prediction}\nReasoning: (Generated reasoning follows below if present)"
        except Exception as e:
            print(f"⚠️ Error on Q{i}: {e}")
            prediction = "Error"

        output_rows.append({
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "subject": subject,
            "category": category,
            "canary": canary,
            "prediction": prediction,
        })

        if i % 20 == 0:
            print(f"✅ Processed {i}/{len(data)} questions...")

    # -----------------------------
    # Save results to CSV
    # -----------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "question", "answer", "answer_type", "subject", "category", "canary", "prediction"
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"✅ All done! Results saved to {args.output}")

if __name__ == "__main__":
    main()
