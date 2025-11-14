# benchmarks/cvqa.py
# Prepares CVQA (afaji/cvqa) into data/benchmarks_data/cvqa.csv + images/
# Note: CVQA test split has no released labels; we set answer="NA".

from datasets import load_dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import json

BENCHMARK_NAME = "cvqa"
HF_DS = "afaji/cvqa"

def _safe_stem(s):
    s = str(s)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)[:120]

def download_and_prepare(data_dir: str) -> str:
    """
    Framework entry point expected by main.py.
    main.py will call: csv_path = download_and_prepare(data_dir)
    Must return the CSV path (string).
    """
    base = Path(data_dir)                 # typically "data/benchmarks_data"
    out_dir = base / BENCHMARK_NAME       # e.g., data/benchmarks_data/cvqa
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base / f"{BENCHMARK_NAME}.csv"

    # CVQA provides "test" split with PIL images + metadata; no labels released.
    # Use a slice for faster testing; change to "test" for full.
    ds = load_dataset(HF_DS, split="test[:200]")

    rows = []
    for ex in ds:
        pil_img: Image.Image = ex["image"]
        q = ex.get("Question") or ex.get("Translated Question") or ""
        subset = ex.get("Subset", "")
        cid = ex.get("ID", "")

        # Options list (we store for reference; not required by the framework schema)
        opts = ex.get("Options") or ex.get("Translated Options") or []

        # Save image with stable name
        fname = f"{_safe_stem(cid or subset)}.png"
        img_rel = f"{BENCHMARK_NAME}/{fname}"              # relative path stored in CSV
        pil_img.save(out_dir / fname)

        # Public test split has no labels; answer="NA"
        rows.append({
            "id": cid or fname,
            "question": q,
            "image_path": img_rel,                         # relative path
            "answer": "NA",                                # no labels available
            # Optional extra columns (allowed by assignment)
            "answer_type": "multipleChoice",
            "category": subset,
            "raw_subject": "",
            "options": json.dumps(opts, ensure_ascii=False)
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)

# Optional convenience for standalone runs:
if __name__ == "__main__":
    p = download_and_prepare("data/benchmarks_data")
    print(f"Saved CSV to {p}")
    print(f"Images in {Path(p).parent / BENCHMARK_NAME}/")
