#!/usr/bin/env python3
"""
MedXpertQA Benchmark Preparation Script

This script prepares the MedXpertQA benchmark data in the required CSV format
for LLM/agentic AI framework evaluation.
"""

import os
import json
import pandas as pd
import shutil
from pathlib import Path


def prepare_medxpertqa_benchmark():
    """
    Prepare MedXpertQA benchmark data and save as CSV with required columns:
    - id: internal id/code for the datapoint
    - question: the question text
    - image_path: relative path to image (if applicable)
    - answer: ground truth answer
    - answer_type: type of answer (multipleChoice/exactMatch/etc)
    - category: subset name (medical_task)
    - raw_subject: subsubset name (body_system)
    """
    
    # Define paths
    input_dir = "eval/data/medxpertqa/input"
    output_dir = "data/benchmarks_data/medxpertqa"
    output_csv = "data/benchmarks_data/medxpertqa.csv"
    
    # Create output directory for images
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both text and multimodal versions
    datasets = [
        ("medxpertqa_text_input.jsonl", "text"),
        ("medxpertqa_mm_input.jsonl", "multimodal")
    ]
    
    all_data = []
    
    for filename, dataset_type in datasets:
        input_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
            
        print(f"Processing {filename}...")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # Extract basic information
                        question_id = data.get('id', f'{dataset_type}_{line_num}')
                        question = data.get('question', '')
                        answer = data.get('label', [])
                        
                        # Convert answer to string format
                        if isinstance(answer, list):
                            answer_str = ', '.join(answer)
                        else:
                            answer_str = str(answer)
                        
                        # Handle images
                        image_path = ""
                        images = data.get('images', [])
                        if images and dataset_type == "multimodal":
                            # Copy images to benchmark directory
                            image_files = []
                            for img in images:
                                if isinstance(img, dict):
                                    img_path = img.get('image_path', '')
                                else:
                                    img_path = str(img)
                                
                                if img_path:
                                    # Source path in eval/images
                                    src_path = os.path.join("eval/images", img_path)
                                    # Destination path in benchmark directory
                                    dst_path = os.path.join(output_dir, img_path)
                                    
                                    if os.path.exists(src_path):
                                        # Copy image to benchmark directory
                                        shutil.copy2(src_path, dst_path)
                                        image_files.append(img_path)
                            
                            if image_files:
                                image_path = ', '.join(image_files)
                        
                        # Extract metadata
                        medical_task = data.get('medical_task', '')
                        body_system = data.get('body_system', '')
                        question_type = data.get('question_type', '')
                        
                        # Determine answer type based on options
                        options = data.get('options', [])
                        if options:
                            answer_type = "multipleChoice"
                        else:
                            answer_type = "exactMatch"
                        
                        # Create row data
                        row_data = {
                            'id': question_id,
                            'question': question,
                            'image_path': image_path,
                            'answer': answer_str,
                            'answer_type': answer_type,
                            'category': medical_task,
                            'raw_subject': body_system
                        }
                        
                        all_data.append(row_data)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {filename}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num} in {filename}: {e}")
                        continue
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        print(f"\nBenchmark preparation completed!")
        print(f"Total questions processed: {len(df)}")
        print(f"CSV saved to: {output_csv}")
        print(f"Images saved to: {output_dir}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Questions with images: {len(df[df['image_path'] != ''])}")
        print(f"- Multiple choice questions: {len(df[df['answer_type'] == 'multipleChoice'])}")
        print(f"- Categories: {df['category'].value_counts().to_dict()}")
        print(f"- Body systems: {df['raw_subject'].value_counts().to_dict()}")
        
        return output_csv
    else:
        print("No data processed!")
        return None


if __name__ == "__main__":
    prepare_medxpertqa_benchmark()
