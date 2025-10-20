"""
SuperGPQA Benchmark Downloader

SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines
A comprehensive benchmark that evaluates graduate-level knowledge and reasoning 
capabilities across 285 disciplines with ~26,529 questions.

Paper: https://arxiv.org/abs/2502.14739
Dataset: https://huggingface.co/datasets/m-a-p/SuperGPQA
"""

import csv
from pathlib import Path
from typing import List
from datasets import load_dataset
from tqdm import tqdm


def download_and_prepare(data_dir: str) -> str:
    """
    Download and prepare SuperGPQA dataset.
    
    This function is called by the main framework and must return the path to the CSV file.
    
    Args:
        data_dir: Directory to save the processed dataset (typically 'data/benchmarks_data')
        
    Returns:
        Path to the saved CSV file
    """
    print("Downloading SuperGPQA dataset from HuggingFace...")
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset("m-a-p/SuperGPQA", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        dataset = load_dataset("m-a-p/SuperGPQA", split="train")
    
    print(f"Dataset loaded: {len(dataset)} questions")
    
    # Define CSV path
    csv_path = data_path / "SuperGPQA.csv"
    
    # Process and save dataset
    _process_and_save_dataset(dataset, csv_path)
    
    print(f"Dataset saved to: {csv_path}")
    return str(csv_path)


def _process_and_save_dataset(dataset, csv_path: Path):
    """Process dataset and save to CSV with required format."""
    
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
        
        for item in tqdm(dataset, desc="Processing questions"):
            # Format question with multiple choice options
            formatted_question = _format_question_with_options(item['question'], item['options'])
            
            row = {
                'id': item['uuid'],                    # Use UUID as internal ID
                'question': formatted_question,        # Question with formatted options (A, B, C, etc.)
                'image_path': '',                      # SuperGPQA is text-only
                'answer': item['answer'],              # Ground truth answer text
                'answer_type': 'multipleChoice',       # All questions are multiple choice
                'category': item['field'],             # Field as category (e.g., "Medicine")
                'raw_subject': item['subfield']        # Subfield as raw_subject (e.g., "Internal Medicine")
            }
            
            writer.writerow(row)


def _format_question_with_options(question: str, options: List[str]) -> str:
    """Format question with labeled multiple choice options (A, B, C, etc.)."""
    # Create labeled options
    formatted_options = []
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        formatted_options.append(f"{letter}. {option}")
    
    # Combine question with options
    return f"{question}\n\n" + "\n".join(formatted_options)


# Optional: Information function (not required by framework but useful)
def get_benchmark_info():
    """Return benchmark information."""
    return {
        'name': 'SuperGPQA',
        'description': 'Graduate-level academic questions across 285 disciplines',
        'paper': 'https://arxiv.org/abs/2502.14739',
        'dataset': 'https://huggingface.co/datasets/m-a-p/SuperGPQA',
        'num_questions': '~26,529',
        'disciplines': 285,
        'question_type': 'Multiple Choice (4-10 options)',
        'difficulty_levels': ['easy', 'middle', 'hard'],
        'domains': [
            'Agronomy', 'Economics', 'Education', 'Engineering', 
            'History', 'Law', 'Literature and Arts', 'Management', 
            'Medicine', 'Military Science', 'Philosophy', 'Science', 'Sociology'
        ]
    }


if __name__ == "__main__":
    # For standalone testing
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperGPQA Benchmark Downloader')
    parser.add_argument('--download', action='store_true', help='Download the dataset')
    parser.add_argument('--data-dir', default='data/benchmarks_data', help='Data directory')
    parser.add_argument('--info', action='store_true', help='Show benchmark information')
    
    args = parser.parse_args()
    
    if args.info:
        info = get_benchmark_info()
        print(f"Benchmark: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Questions: {info['num_questions']}")
        print(f"Disciplines: {info['disciplines']}")
        print(f"Paper: {info['paper']}")
    
    if args.download:
        csv_path = download_and_prepare(args.data_dir)
        print(f"Download completed: {csv_path}")