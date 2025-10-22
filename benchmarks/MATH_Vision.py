"""
MATH-Vision Benchmark Preparation Script
========================================

Author: Shengtao Gao
Course: EAI 6080 (Sinitskiy, Anton, Northeastern University)
File Path: benchmarks/MATH_Vision.py

----------------------------------------------------------
Purpose:
----------------------------------------------------------
This script prepares the MATH-Vision (MATH-V) benchmark dataset
for use within the unified benchmarking and evaluation framework
developed in this course.

It performs the following tasks:
  1. Automatically downloads the MATH-Vision dataset from Hugging Face.
  2. Downloads and extracts associated images (images.zip).
  3. Converts the dataset into a standardized CSV format compatible
      with our evaluation pipeline.
  4. Saves the processed data to:
         data/benchmarks_data/MATH_Vision/MATH_Vision.csv

----------------------------------------------------------
Important Notes:
----------------------------------------------------------
• The current Hugging Face dataset ("MathLLMs/MathVision")
  contains **only the test split** (~3,040 samples).
  It is provided by the original authors for benchmarking purposes.

• The **complete dataset (train/validation/test)** is not publicly
  released at the time of writing. To obtain the full dataset,
  users would need access from the MathLLMs research team or
  follow updates on:
      https://huggingface.co/datasets/MathLLMs/MathVision

• This script follows the structure of benchmarks/HLE.py as required
  in HW5 Part A. Please do not modify other repository files.

----------------------------------------------------------
Expected Output:
----------------------------------------------------------
The resulting CSV file will contain the following columns:
    id, question, image_path, answer

----------------------------------------------------------

"""


import os
import zipfile
import requests
from pathlib import Path
from typing import Union, Optional

import pandas as pd
from datasets import load_dataset

try:
    import local_secrets  # type: ignore
    HF_TOKEN: Optional[str] = getattr(local_secrets, "HUGGINGFACE_TOKEN", None) or None
except Exception:
    HF_TOKEN = None


def _download_file(url: str, dst: Path, hf_token: Optional[str] = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def download_and_prepare(data_dir: Union[str, Path]) -> Optional[Path]:

    data_dir = Path(data_dir)
    bench_root = data_dir  # main.py 已把 data_dir 指到 data/benchmarks_data/
    bench_name = "MATH_Vision"
    bench_dir = bench_root / bench_name
    img_zip = bench_dir / "images.zip"
    img_dir = bench_dir / "images"
    csv_path = bench_root / f"{bench_name}.csv"

    try:
        if not img_dir.exists():
            if not img_zip.exists():
                print(" Downloading images.zip from Hugging Face...")
                zip_url = "https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/images.zip"
                _download_file(zip_url, img_zip, HF_TOKEN)
            print(" Extracting images.zip ...")
            with zipfile.ZipFile(img_zip, "r") as zf:
                zf.extractall(img_dir)
            print(" Images extracted:", img_dir)

        print(" Loading dataset metadata (test split only)...")
        ds = load_dataset("MathLLMs/MathVision", token=HF_TOKEN)["test"]

        rows = []
        for item in ds:

            image_rel = os.path.basename(item["image"])
            rows.append({
                "id": item["id"],
                "question": item["question"],
                "image_path": f"{bench_name}/images/{image_rel}",
                "answer": item["answer"],
            })

        df = pd.DataFrame(rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(" Benchmark CSV saved:", csv_path)

        return csv_path

    except Exception as e:
        print(f" MATH_Vision preparation failed: {e}")
        return None