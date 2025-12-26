import json
import pandas as pd
from pathlib import Path

def download_and_prepare(data_dir=None):
    """
    Convert RAG-QA Science JSONL to standardized CSV format
    """
    if data_dir is None:
        data_dir = Path("data/benchmarks_data")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "RAG_QA_Arena_Science.csv"
    
    # Read JSONL file (adjust path if needed)
    jsonl_path = Path("small_rag_qa_test.jsonl")  # Use your test file
    
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        return None
    
    try:
        # Read JSONL and convert to list of dicts
        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_cols = ['id', 'question', 'answer', 'answer_type', 'category', 'raw_subject']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in data")
        
        # Add image_path column (RAG-QA doesn't have images)
        if 'image_path' not in df.columns:
            df['image_path'] = None
        
        # Reorder columns
        cols_order = ['id', 'question', 'image_path', 'answer', 'answer_type', 'category', 'raw_subject']
        df = df[[col for col in cols_order if col in df.columns]]
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Successfully created: {csv_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        return csv_path
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    path = download_and_prepare()
    if path:
        print(f"\n✅ CSV file saved to: {path}")
