# Benchmark Preparation System

This system provides a unified interface for preparing and running benchmarks for LLM/agentic AI framework evaluation.

## Structure

```
new_benchmark_system/
├── benchmarks/                    # Benchmark preparation scripts
│   └── medxpertqa.py            # MedXpertQA benchmark preparation
├── data/benchmarks_data/        # Prepared benchmark data
│   └── medxpertqa/             # MedXpertQA images
│   └── medxpertqa.csv          # MedXpertQA benchmark CSV
├── outputs/benchmark_predictions/ # Model predictions
├── main1.py                    # Main script for running benchmarks
└── BENCHMARK_README.md         # This documentation
```

## Usage

Navigate to the new_benchmark_system directory first:
```bash
cd new_benchmark_system
```

### Prepare a benchmark
```bash
python main1.py --download --benchmarks medxpertqa
```

### Run predictions
```bash
python main1.py --download --benchmarks medxpertqa --predict --models gpt-4o-mini
```

### List available benchmarks
```bash
python main1.py --list-benchmarks
```

### List available models
```bash
python main1.py --list-models
```

## Benchmark CSV Format

The prepared benchmark CSV files contain the following columns:

- `id`: Internal ID/code for the datapoint
- `question`: The question text
- `image_path`: Relative path to image (if applicable)
- `answer`: Ground truth answer
- `answer_type`: Type of answer (multipleChoice/exactMatch/etc)
- `category`: Subset name (e.g., medical_task)
- `raw_subject`: Subsubset name (e.g., body_system)

## Adding New Benchmarks

To add a new benchmark:

1. Create a new file `benchmarks/{your_benchmark_name}.py`
2. Implement a function `prepare_{your_benchmark_name}_benchmark()` that:
   - Loads the benchmark data
   - Processes images (if applicable)
   - Saves data as CSV in `data/benchmarks_data/{your_benchmark_name}.csv`
   - Returns the CSV path on success

## Example: MedXpertQA

The MedXpertQA benchmark includes:
- 4,450 total questions
- 2,000 questions with images (multimodal)
- 2,450 text-only questions
- Categories: Diagnosis (2,249), Treatment (1,194), Basic Science (1,007)
- Body systems: Skeletal, Cardiovascular, Nervous, etc.

All questions are multiple choice format with ground truth answers.
