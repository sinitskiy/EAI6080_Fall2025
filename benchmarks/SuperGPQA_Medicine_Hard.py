"""
SuperGPQA Medicine Hard Benchmark Downloader

Filtered subset of SuperGPQA containing only Medicine discipline questions 
with Hard difficulty level. This represents the most challenging medical 
graduate-level questions from the SuperGPQA dataset.

Paper: https://arxiv.org/abs/2502.14739
Dataset: https://huggingface.co/datasets/m-a-p/SuperGPQA
"""

import csv
from pathlib import Path
from typing import List
from datasets import load_dataset
from tqdm import tqdm


def download_and_prepare(data_dir):
    """
    Download and prepare SuperGPQA Medicine Hard subset.
    
    This function downloads the full SuperGPQA dataset and filters it to include
    only Medicine discipline questions with Hard difficulty level.
        
    Returns:
        Path to the saved CSV file
    """
    print("Downloading SuperGPQA dataset from HuggingFace...")
    
    # check for existing data file
    csv_path = data_dir / "SuperGPQA_Medicine_Hard.csv"
    if csv_path.exists():
        return csv_path
    
    # Load full dataset from HuggingFace
    try:
        dataset = load_dataset("m-a-p/SuperGPQA", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        dataset = load_dataset("m-a-p/SuperGPQA", split="train")
    
    print(f"Full dataset loaded: {len(dataset)} questions")
    
    # Filter for Medicine discipline and Hard difficulty
    filtered_dataset = []
    for item in dataset:
        if item['discipline'] == 'Medicine' and item['difficulty'] == 'hard':
            filtered_dataset.append(item)
    
    print(f"Filtered to Medicine Hard questions: {len(filtered_dataset)} questions")
    
    if len(filtered_dataset) == 0:
        print("Warning: No Medicine Hard questions found in dataset!")
        return None
    
    # Process and save filtered dataset
    _process_and_save_dataset(filtered_dataset, csv_path)
    
    print(f"Dataset saved to: {csv_path}")
    return csv_path


def _process_and_save_dataset(dataset, csv_path: Path):
    """Process filtered dataset and save to CSV with required format."""
    
    # Required columns as specified in assignment
    fieldnames = [
        'id',           # internal id/code for the given datapoint
        'question',     # the question text
        'image_path',   # path to image (empty for SuperGPQA - text only)
        'answer',       # ground truth answer
        'answer_type',  # multipleChoice (all SuperGPQA questions)
        'category',     # subset name (field)
        'raw_subject'   # subsubset name (subfield)
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in tqdm(dataset, desc="Processing Medicine Hard questions"):
            # Format question with multiple choice options
            formatted_question, answer_letter = _format_question_with_options(item)
            
            row = {
                'id': item['uuid'],                    # Use UUID as internal ID
                'question': formatted_question,        # Question with formatted options (A, B, C, etc.)
                'image_path': '',                      # SuperGPQA is text-only
                'answer': answer_letter,               # Ground truth answer (letter)
                'answer_type': 'multipleChoice',       # All questions are multiple choice
                'category': item['field'],             # Field as category (should be Medicine-related)
                'raw_subject': item['subfield']        # Medical subfield (e.g., "Internal Medicine")
            }
            
            writer.writerow(row)


def _format_question_with_options(item: dict) -> tuple[str, str]:
    """Format question with labeled multiple choice options (A, B, C, etc.) and return the correct answer letter."""
    # Create labeled options
    formatted_options = []
    answer_letter = None
    for i, option in enumerate(item['options']):
        letter = chr(ord('A') + i)
        formatted_options.append(f"{letter}. {option}")
        if option == item['answer']:
            answer_letter = letter

    # Make sure we found the correct answer letter, otherwise use answer_letter from the item
    if answer_letter is None:
        print(f"Warning: Correct answer not found among options for question ID {item['uuid']}. Using provided answer_letter instead: {item['answer_letter']}.")
        answer_letter = item['answer_letter']
    # Combine question with options
    return f"{item['question']}\n\n" + "\n".join(formatted_options), answer_letter


def get_benchmark_info():
    """Return benchmark information."""
    return {
        'name': 'SuperGPQA_Medicine_Hard',
        'description': 'Hard difficulty medical questions from SuperGPQA dataset',
        'paper': 'https://arxiv.org/abs/2502.14739',
        'dataset': 'https://huggingface.co/datasets/m-a-p/SuperGPQA',
        'discipline': 'Medicine',
        'difficulty': 'Hard',
        'question_type': 'Multiple Choice (4-10 options)',
        'subset_criteria': 'discipline == Medicine AND difficulty == hard',
        'medical_domains': [
            'Internal Medicine', 'Surgery', 'Pediatrics', 'Obstetrics and Gynecology',
            'Pathology', 'Pharmacology', 'Clinical Medicine', 'Basic Medicine'
        ]
    }


if __name__ == "__main__":
    # For standalone testing
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperGPQA Medicine Hard Benchmark Downloader')
    parser.add_argument('--download', action='store_true', help='Download the dataset')
    parser.add_argument('--info', action='store_true', help='Show benchmark information')
    
    args = parser.parse_args()
    
    if args.info:
        info = get_benchmark_info()
        print(f"Benchmark: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Discipline: {info['discipline']}")
        print(f"Difficulty: {info['difficulty']}")
        print(f"Subset criteria: {info['subset_criteria']}")
        print(f"Paper: {info['paper']}")
    
    if args.download:
        p = download_and_prepare()
        print(p)
