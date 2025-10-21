import os
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd


def download_and_prepare(data_dir=None):
    if data_dir is None:
        data_dir = Path("./mathvision_data")
    else:
        data_dir = Path(data_dir)
        
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "MathVision.csv"
    if csv_path.exists():
        return csv_path

    try:
        # Load the dataset from huggingface
        if os.getenv("HF_TOKEN"):
            login(token=os.getenv("HF_TOKEN"))
        
        # Load the dataset
        ds = load_dataset("MathLLMs/MathVision")
        
        # Create a folder with images
        image_dir = data_dir / "data/benchmark_data/MathVision"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert dataset to pandas DataFrame
        test_dataset = ds['test']
        df_data = []
        
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            df_data.append({
                'id': i+1,
                'question': sample.get('question', ''),
                'options': sample.get('options', ''),
                'answer': sample.get('answer', ''),
                'level': sample.get('level', ''),
                'subject': sample.get('subject', ''),
                'image_path': None
            })
        
        df = pd.DataFrame(df_data)
        
        # Save images and add image paths to dataframe
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            image_data = sample.get('decoded_image', None)
            
            if image_data is not None:
                try:
                    # Process the image data
                    if isinstance(image_data, list) and len(image_data) > 0:
                        image_data = image_data[0]
                    
                    if isinstance(image_data, Image.Image):
                        # If it's already a PIL Image, save it directly
                        img_path = image_dir / f"{i}.png"
                        image_data.save(img_path)
                        df.loc[i, 'image_path'] = str(img_path)
                    elif isinstance(image_data, bytes):
                        # If it's bytes, open and save
                        img = Image.open(BytesIO(image_data))
                        img_path = image_dir / f"{i}.png"
                        img.save(img_path)
                        df.loc[i, 'image_path'] = str(img_path)
                    else:
                        print(f"Unknown image format for sample {i}: {type(image_data)}")
                        
                except Exception as e:
                    print(f"Error processing image for sample {i}: {e}")
        
        # Save the dataset as a csv file
        df.to_csv(csv_path, index=False)
        print(f"Successfully processed {len(df)} samples")
        print(f"Images saved to: {image_dir}")
        print(f"CSV saved to: {csv_path}")
        
        return csv_path
    
    except Exception as e:
        print(f"MathVision: download failed: {e}")
        return None


if __name__ == "__main__":
    p = download_and_prepare()
    print(p)