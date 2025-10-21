import os
from pathlib import Path
import zipfile
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def download_and_prepare(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "MedXpertQA_MM.csv"
    if csv_path.exists():
        return csv_path

    try:
        # load the dataset from huggingface
        ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", split="test")
        keep = ['id', 'question','options','label','images','medical_task','body_system','question_type']
        df = ds.remove_columns([c for c in ds.column_names if c not in keep]).to_pandas()
        
        # create a folder with images
        image_dir = data_dir /"MedXpertQA_MM"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        zip_path = hf_hub_download(
                repo_id="TsinghuaC3I/MedXpertQA",
                filename="images.zip",
                repo_type="dataset",
                local_dir=image_dir,
            )
        
        print(f"✓ Downloaded to: {zip_path}")
        print(f"\nExtracting...")

# Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_dir)

        print(f"✓ Extraction complete!")
        
        # Move images from nested folder to parent folder
        extracted_images_dir = image_dir / "images"
        if extracted_images_dir.exists():
            import shutil
            for img_file in extracted_images_dir.iterdir():
                shutil.move(str(img_file), str(image_dir / img_file.name))
            # Remove the now-empty images folder
            extracted_images_dir.rmdir()
            print(f"✓ Moved images directly to {image_dir}")
        
        # Optional: Remove zip file
        os.remove(zip_path)
        print(f"✓ Cleaned up zip file")      
        
        
        # save images from ds['image'] and add a column to df with paths to these images
        image_paths = []
        
        for i, img in enumerate(ds['images']):
            if img and img != '':
                # If img is a single filename
                if isinstance(img, str):
                    # Reference the images folder (now directly in image_dir)
                    img_path = image_dir / img
                    image_paths.append([str(img_path)])
                # If img is a list of filenames (multiple images)
                elif isinstance(img, list):
                    # Store as list of image paths
                    img_paths = [str(image_dir / im) for im in img if im]
                    image_paths.append(img_paths)
            else:
                image_paths.append([])
        
        df['image_path'] = image_paths

        # save the dataset as a csv file
        df.to_csv(csv_path, index=False)
        return csv_path
    
    except Exception as e:
        print(f"MedXpertQA_MM: download failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    p = download_and_prepare(data_dir="data/benchmarks_data/")
    print(p)