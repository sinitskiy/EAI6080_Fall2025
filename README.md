# EAI6080_Fall2025 - LLM Benchmarking Project

Course project at Northeastern University in EAI 6080, Fall 2025

## Overview

This project automates benchmarking of various LLMs and Agentic AI systems across multiple evaluation benchmarks. The system follows a modular architecture where each benchmark and model is implemented in its own file, making it easy to add new benchmarks or models.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sinitskiy/EAI6080_Fall2025.git
cd EAI6080_Fall2025
```

2. Create and activate a virtual environment:

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys as environment variables:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:GOOGLE_API_KEY="your-google-api-key"

# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

> **Note:** Always activate your virtual environment before running scripts:
> - Windows: `.\.venv\Scripts\Activate.ps1`
> - Linux/Mac: `source .venv/bin/activate`
> 
> Your prompt should show `(.venv)` when the virtual environment is active.

## Usage

### Run Everything (Full Pipeline)

```bash
python main.py --all
```

This runs all four steps:
1. Download benchmarks
2. Run model predictions
3. Evaluate correctness
4. Generate summary table

### Run Individual Steps

```bash
# Step 1: Download benchmarks only
python main.py --download

# Step 2: Run predictions only
python main.py --predict

# Step 3: Evaluate predictions only
python main.py --evaluate

# Step 4: Generate summary only
python main.py --summary
```

### Run Specific Benchmarks or Models

```bash
# Run specific benchmarks
python main.py --download --benchmarks HLE BixBench

# Run specific models
python main.py --predict --models GPT_5_mini

# Combine filters
python main.py --all --benchmarks HLE --models GPT_5_mini
```

### View Help

```bash
python main.py --help
```

## Adding New Benchmarks

1. Create a new file in `benchmarks/` (e.g., `benchmarks/MyBenchmark.py`)
2. Implement the `download_benchmark()` function that returns a pandas DataFrame
3. The DataFrame must have these columns:
   - `question_id`: Unique identifier
   - `question_text`: The question
   - `ground_truth`: Correct answer
   - `subset`: Category (optional)
   - `metadata`: Additional info (optional)
   - `image_path`: For multimodal benchmarks (optional)

See `benchmarks/README.md` for a template.

## Adding New Models

1. Create a new file in `models/` (e.g., `models/MyModel.py`)
2. Implement two required functions:
   - `initialize_model()`: Set up and return the model
   - `run_predictions(model, csv_path, model_name)`: Run predictions on a CSV

See `models/README.md` for a template.

## Adding Custom Evaluators

By default, all benchmarks use `evaluators/default_evaluator.py` which implements:
- Exact text matching
- Multiple choice answer extraction (handles "B", "B.", "B) because...")
- Semantic similarity matching

To create a custom evaluator for a specific benchmark:

1. Create `evaluators/{benchmark_name}_evaluator.py`
2. Implement the `evaluate(csv_path)` function

See `evaluators/README.md` for details.

## Benchmarks Included

1. **HLE** - Healthcare and Life Sciences Evaluation (all subsets)
2. **hle-gold-bio-chem** - Biology and Chemistry subset
3. **BixBench**
4. **SuperGPQA Medicine Hard**
5. **HealthBench Hard**
6. **MedXpertQA** (Text and Multimodal)
7. **MATH-Vision**
8. **CVQA**
9. **LitQA2**
10. **RAG-QA Arena Science**

## Models Included

### API-Based
- **GPT-5-mini** (OpenAI)
- **gemini-2.5-pro** (Google)

### Local Models
- **Qwen2.5-VL-7B-Instruct** (Vision-Language)
- **Qwen/Qwen2.5-14B-Instruct**
- **DeepSeek-R1-Distill-Qwen-7B**

## Output

Results are saved in `data/results/`:
- `summary_table.csv` - Accuracy scores for all models and benchmarks
- `detailed_results.csv` - Per-question breakdown
- `summary_report.md` - Human-readable markdown report

Benchmark CSVs with predictions are saved in `data/benchmarks/{benchmark_name}/questions.csv`

## Development Workflow

1. **Download benchmarks** once (they're cached locally)
2. **Run predictions** incrementally (existing predictions are skipped)
3. **Evaluate** after all predictions are complete
4. **Generate summary** to see final results

Progress is logged to both console and `benchmarking.log`.

## Platform Support

This project supports multiple deployment environments:
- Windows laptops (local development with optional GPU)
- Mac laptops (Apple Silicon M1/M2/M3/M4 with MPS)
- Northeastern University HPC cluster (SLURM with NVIDIA A100 GPUs)

See `manual_4_LLM_installation.md` for platform-specific setup instructions.

## Notes

- **Data files are not uploaded to GitHub** (excluded in `.gitignore`)
- Predictions are saved incrementally, so interrupted runs can be resumed
- API rate limits are handled with retries and exponential backoff
- Large models may require quantization for local deployment
