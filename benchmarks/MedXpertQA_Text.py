from pathlib import Path
from datasets import load_dataset


def download_and_prepare(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "MedXpertQA_Text.csv"
    if csv_path.exists():
        return csv_path

    try:
        # load the dataset from huggingface
        ds = load_dataset("TsinghuaC3I/MedXpertQA", 'Text', split="test")
        keep = ['id', 'question', 'label', 'medical_task', 'body_system',
                # 'question_type'
                ]
        df = ds.remove_columns([c for c in ds.column_names if c not in keep]).to_pandas()
        df = df.rename(columns={'label':'answer', 'medical_task':'category', 'body_system':'raw_subject'})
        df['answer_type'] = 'multiple_choice'
        
        # save the dataset as a csv file
        df.to_csv(csv_path, index=False)
        return csv_path
    
    except Exception as e:
        print(f"MedXpertQA_Text: download failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    p = download_and_prepare(data_dir="../data/benchmarks_data/")
    print(p)