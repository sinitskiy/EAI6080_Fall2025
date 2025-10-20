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
            print(f"BixBench: already exists at {csv_path.resolve()}")
            return csv_path

        print(" Downloading full BixBench dataset from Hugging Face (futurehouse/BixBench)...")
        ds = load_dataset("futurehouse/BixBench")

        # combine all splits into one DataFrame
        dfs = []
        for name, split in ds.items():
            df = pd.DataFrame(split)
            df["split"] = name
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # keep all key fields needed for benchmarking
        keep = [
            "id", "question", "ideal", "hypothesis", "result",
            "distractors", "categories", "eval_mode", "data_folder",
            "answer", "tag", "version"
        ]
        df = df[[c for c in keep if c in df.columns]]

        # save to CSV
        df.to_csv(csv_path, index=False)
        print(f" Download complete. Saved to: {csv_path.resolve()}")
        print(f" Total rows: {len(df)} | Columns: {list(df.columns)}")
        return csv_path

    except Exception as e:
        print(f" BixBench: download or save failed. Error: {e}")
        return None


if __name__ == "__main__":
    download_and_prepare()
