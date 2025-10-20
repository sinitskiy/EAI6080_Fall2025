import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def download_and_prepare():
    try:
        # set path to repo's data folder (updated to benchmarks_data)
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data" / "benchmarks_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / "BixBench.csv"

        # skip download if file already exists
        if csv_path.exists():
            print(f" BixBench: already exists at {csv_path.resolve()}")
            return csv_path

        print(" Downloading BixBench dataset from Hugging Face (futurehouse/BixBench)...")
        ds = load_dataset("futurehouse/BixBench")

        # combine all splits into one DataFrame
        dfs = []
        for name, split in ds.items():
            df = pd.DataFrame(split)
            df["split"] = name
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # --- Map fields to HLE-compatible format ---
        df_final = pd.DataFrame()
        df_final["id"] = df["id"]
        df_final["question"] = df["question"]
        df_final["answer"] = df["ideal"]                      # use real answer text
        df_final["answer_type"] = df["eval_mode"]             # evaluation type
        df_final["raw_subject"] = df["categories"]            # broad subject area
        df_final["category"] = df["tag"]                      # subset / benchmark name
        df_final["image_path"] = ""                           # no images in BixBench

        # --- Save standardized CSV ---
        df_final.to_csv(csv_path, index=False)
        print(f" Download complete. Saved standardized BixBench at: {csv_path.resolve()}")
        print(f" Total rows: {len(df_final)} | Columns: {list(df_final.columns)}")
        print(" BixBench dataset verified and ready for benchmarking.")
        return csv_path

    except Exception as e:
        print(f" BixBench: download or save failed. Error: {e}")
        return None


if __name__ == "__main__":
    download_and_prepare()
