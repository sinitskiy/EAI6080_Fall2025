from pathlib import Path
from datasets import load_dataset


def download_and_prepare(data_dir):
    csv_path = data_dir / "hle-gold-bio-chem.csv"
    if csv_path.exists():
        return csv_path

    try:
        # load the dataset from huggingface
        ds = load_dataset("futurehouse/hle-gold-bio-chem", split="train")
        keep = ['id', 'question', 'answer', 'answer_type', 'raw_subject', 'category']
        df = ds.remove_columns([c for c in ds.column_names if c not in keep]).to_pandas()
        
        # save the dataset as a csv file
        df.to_csv(csv_path, index=False)
        return csv_path
    
    except Exception as e:
        print(f"hle-gold-bio-chem: download failed: {e}")
        return None


if __name__ == "__main__":
    p = download_and_prepare()
    print(p)
