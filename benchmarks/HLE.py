import pandas as pd
from pathlib import Path
from datasets import load_dataset

def download_and_prepare(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "HLE.csv"

    if csv_path.exists():
        return csv_path

    try:
        # Load the dataset using the datasets library
        dataset = load_dataset("cais/hle", split="test")
        df = pd.DataFrame(dataset)

        # Save the dataset to a CSV file
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f"Error loading HLE dataset: {e}")
        return None

if __name__ == "__main__":
    data_dir = "../data"
    out = download_and_prepare(data_dir)
    print(out)