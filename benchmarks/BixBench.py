import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import shutil
import requests


def download_and_prepare(data_dir=None):
    try:
        # === PATH SETUP ===
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data" / "benchmarks_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Folder to store .zip data files
        data_files_dir = data_dir / "BixBench_datafiles"
        data_files_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "BixBench.csv"

        # === SKIP IF ALREADY EXISTS ===
        if csv_path.exists():
            print(f"📁 BixBench: already exists at {csv_path.resolve()}")
            return csv_path

        print("📘 Downloading BixBench dataset from Hugging Face (futurehouse/BixBench)...")
        ds = load_dataset("futurehouse/BixBench")
        print(" Dataset metadata loaded successfully")

        # === COMBINE SPLITS INTO ONE DF ===
        dfs = []
        for name, split in ds.items():
            df = pd.DataFrame(split)
            df["split"] = name
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        print(f" Combined dataset. Total rows: {len(df)}")

        # === DOWNLOAD CAPSULEFOLDER FILES ===
        def fetch_data_file(data_folder_name):
            """Download or reuse CapsuleFolder zip files."""
            if not data_folder_name:
                return ""
            file_name = os.path.basename(data_folder_name)
            dest_path = data_files_dir / file_name

            # Skip if already exists
            if dest_path.exists():
                return str(dest_path.relative_to(repo_root))

            # Try multiple possible URLs
            potential_urls = [
                f"https://huggingface.co/datasets/futurehouse/BixBench/resolve/main/{file_name}",
                f"https://huggingface.co/datasets/futurehouse/BixBench/resolve/main/data/{file_name}",
            ]

            for url in potential_urls:
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    if r.status_code == 200:
                        with open(dest_path, "wb") as f:
                            shutil.copyfileobj(r.raw, f)
                        print(f" Downloaded: {file_name}")
                        return str(dest_path.relative_to(repo_root))
                except Exception as e:
                    print(f" Error downloading {file_name} from {url}: {e}")

            print(f" Skipped (not found): {file_name}")
            return ""

        print("📦 Downloading all CapsuleFolder zip files (this may take time)...")
        df["data_path"] = df["data_folder"].apply(fetch_data_file)

        # === MAP TO HLE-COMPATIBLE FORMAT ===
        df_final = pd.DataFrame()
        df_final["id"] = df["id"]
        df_final["question"] = df["question"]
        df_final["answer"] = df["ideal"]               # correct answer text
        df_final["answer_type"] = df["eval_mode"]      # evaluation type
        df_final["raw_subject"] = df["categories"]     # domain categories
        df_final["category"] = df["tag"]               # benchmark/subset name
        df_final["data_path"] = df["data_path"]        # linked data zip files

        # === SAVE CSV ===
        df_final.to_csv(csv_path, index=False)
        print(f"\n Download complete. Saved standardized BixBench at: {csv_path.resolve()}")
        print(f" Total rows: {len(df_final)} | Columns: {list(df_final.columns)}")
        print(f" All CapsuleFolder zips saved in: {data_files_dir}")
        print(" BixBench dataset verified and ready for benchmarking.")
        return csv_path

    except Exception as e:
        print(f"❌ BixBench: download or save failed. Error: {e}")
        return None


if __name__ == "__main__":
    download_and_prepare()
