import os
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login


def download_and_prepare(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "HLE.csv"
    if csv_path.exists():
        return csv_path

    try:
        # load the dataset from huggingface
        login(token=os.getenv("HF_TOKEN"))
        ds = load_dataset("cais/hle", split="test")
        keep = ['id', 'question', 'answer', 'answer_type', 'raw_subject', 'category']
        df = ds.remove_columns([c for c in ds.column_names if c not in keep]).to_pandas()
        
        # create a folder with images
        image_dir = data_dir / "HLE_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # save images from ds['image'] and add a column to df with paths to these images
        df['image_path'] = None
        for i, img in enumerate(ds['image']):
            if img != '':
                img_data = base64.b64decode(img.split(',')[1])
                img1 = Image.open(BytesIO(img_data))
                img_path = image_dir / f"{i}.png"
                img1.save(img_path)
                df.loc[i, 'image_path'] = str(img_path)

        # save the dataset as a csv file
        df.to_csv(csv_path, index=False)
        return csv_path
    
    except Exception as e:
        print(f"HLE: download failed: {e}")
        return None


if __name__ == "__main__":
    p = download_and_prepare()
    print(p)
