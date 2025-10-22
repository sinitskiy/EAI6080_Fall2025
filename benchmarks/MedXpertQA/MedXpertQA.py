#!/usr/bin/env python3
"""
Main script for running benchmark preparation and prediction

This script provides a unified interface for:
1. Downloading/preparing benchmark data
2. Running predictions on prepared benchmarks
3. Supporting multiple models and benchmarks
"""

import os
import sys
import argparse
import importlib
import pandas as pd
from pathlib import Path


def download_benchmark(benchmark_name):
    """
    Download/prepare benchmark data using the benchmark-specific script
    
    Args:
        benchmark_name (str): Name of the benchmark to prepare
    """
    print(f"Preparing benchmark: {benchmark_name}")
    
    # Import the benchmark preparation script
    try:
        benchmark_module = importlib.import_module(f"benchmarks.{benchmark_name}")
        if hasattr(benchmark_module, 'prepare_medxpertqa_benchmark'):
            # Call the preparation function
            result = benchmark_module.prepare_medxpertqa_benchmark()
            if result:
                print(f"[SUCCESS] Benchmark {benchmark_name} prepared successfully")
                return True
            else:
                print(f"[ERROR] Failed to prepare benchmark {benchmark_name}")
                return False
        else:
            print(f"[ERROR] Benchmark {benchmark_name} does not have a preparation function")
            return False
    except ImportError as e:
        print(f"[ERROR] Could not import benchmark {benchmark_name}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error preparing benchmark {benchmark_name}: {e}")
        return False


def run_prediction(benchmark_name, model_name):
    """
    Run prediction on the prepared benchmark using the specified model
    
    Args:
        benchmark_name (str): Name of the benchmark
        model_name (str): Name of the model to use
    """
    print(f"Running prediction on {benchmark_name} with model {model_name}")
    
    # Check if benchmark data exists
    benchmark_csv = f"data/benchmarks_data/{benchmark_name}.csv"
    if not os.path.exists(benchmark_csv):
        print(f"[ERROR] Benchmark data not found: {benchmark_csv}")
        print("Please run with --download first")
        return False
    
    # Load benchmark data
    try:
        df = pd.read_csv(benchmark_csv)
        print(f"Loaded {len(df)} questions from {benchmark_csv}")
    except Exception as e:
        print(f"[ERROR] Error loading benchmark data: {e}")
        return False
    
    # For now, we'll create a simple prediction framework
    # In a real implementation, this would integrate with the model inference system
    print(f"Running predictions with {model_name}...")
    
    # Create output directory
    output_dir = f"outputs/benchmark_predictions/{model_name}/{benchmark_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulate predictions (in real implementation, this would call the model)
    predictions = []
    for idx, row in df.iterrows():
        # This is a placeholder - in reality, you would:
        # 1. Load the model
        # 2. Process the question (and image if present)
        # 3. Generate prediction
        # 4. Store results
        
        prediction = {
            'id': row['id'],
            'question': row['question'],
            'ground_truth': row['answer'],
            'prediction': f"Model prediction for {row['id']}",  # Placeholder
            'correct': False,  # Placeholder
            'model': model_name,
            'benchmark': benchmark_name
        }
        predictions.append(prediction)
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    output_file = os.path.join(output_dir, f"{benchmark_name}_predictions.csv")
    predictions_df.to_csv(output_file, index=False)
    
    print(f"[SUCCESS] Predictions saved to: {output_file}")
    print(f"Total predictions: {len(predictions)}")
    
    return True


def list_available_benchmarks():
    """List all available benchmarks"""
    benchmarks_dir = "benchmarks"
    if not os.path.exists(benchmarks_dir):
        print("No benchmarks directory found")
        return []
    
    benchmarks = []
    for file in os.listdir(benchmarks_dir):
        if file.endswith('.py') and file != '__init__.py':
            benchmark_name = file[:-3]  # Remove .py extension
            benchmarks.append(benchmark_name)
    
    return benchmarks


def list_available_models():
    """List all available models"""
    # This would typically read from a config file or model registry
    # For now, return some example models
    return [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "gemini-1.5-pro",
        "deepseek-chat"
    ]


def main():
    parser = argparse.ArgumentParser(description="Benchmark preparation and prediction system")
    parser.add_argument("--download", action="store_true", help="Download/prepare benchmark data")
    parser.add_argument("--benchmarks", type=str, help="Benchmark name to prepare")
    parser.add_argument("--predict", action="store_true", help="Run predictions")
    parser.add_argument("--models", type=str, help="Model name to use for prediction")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available benchmarks")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_benchmarks:
        benchmarks = list_available_benchmarks()
        print("Available benchmarks:")
        for benchmark in benchmarks:
            print(f"  - {benchmark}")
        return
    
    if args.list_models:
        models = list_available_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")
        return
    
    # Validate arguments
    if args.download and not args.benchmarks:
        print("Error: --benchmarks required when using --download")
        return
    
    if args.predict and not args.models:
        print("Error: --models required when using --predict")
        return
    
    if not args.download and not args.predict:
        print("Error: Please specify --download and/or --predict")
        return
    
    # Run benchmark preparation
    if args.download:
        success = download_benchmark(args.benchmarks)
        if not success:
            print("Benchmark preparation failed")
            return
    
    # Run predictions
    if args.predict:
        success = run_prediction(args.benchmarks, args.models)
        if not success:
            print("Prediction failed")
            return
    
    print("[SUCCESS] All operations completed successfully!")


if __name__ == "__main__":
    main()
