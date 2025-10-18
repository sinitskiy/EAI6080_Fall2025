import argparse
import importlib
import importlib.util
# from importlib.machinery import SourceFileLoader
from pathlib import Path
import pandas as pd
import local_secrets

def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    raise ImportError(f"Cannot load module {module_name} from {file_path}")


def load_benchmarks():
    benchmarks = {}
    benchmark_dir = Path(__file__).parent / "benchmarks"
    if not benchmark_dir.exists():
        return benchmarks
    
    for file in benchmark_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        safe_name = file.stem.replace("-", "_")
        module = _load_module_from_path(f"benchmarks.{safe_name}", file)
        if hasattr(module, "download_and_prepare"):
            benchmarks[file.stem] = module.download_and_prepare
    return benchmarks

def load_models():
    models = {}
    model_dir = Path(__file__).parent / "models"
    if not model_dir.exists():
        return models
    
    for file in model_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        safe_name = file.stem.replace("-", "_")
        module = _load_module_from_path(f"models.{safe_name}", file)
        if hasattr(module, "predict"):
            models[file.stem] = module.predict
    return models

def load_evaluators():
    evaluators = {}
    eval_dir = Path(__file__).parent / "evaluators"
    if not eval_dir.exists():
        return evaluators
    
    for file in eval_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        safe_name = file.stem.replace("-", "_")
        module = _load_module_from_path(f"evaluators.{safe_name}", file)
        if hasattr(module, "evaluate"):
            key = file.stem
            if key.endswith("_eval"):
                key = key[: -len("_eval")]
            evaluators[key] = module.evaluate
    return evaluators

def prepare_benchmarks(benchmarks, data_dir, selected=None):
    data_files = {}
    for name, func in benchmarks.items():
        if selected and name not in selected:
            continue
        print(f"Preparing benchmark: {name}")
        csv_path = func(data_dir)
        try:
            path_obj = Path(csv_path) if csv_path else None
        except TypeError:
            path_obj = None
        if path_obj and path_obj.exists():
            data_files[name] = path_obj
        else:
            print(f"  Skipping {name} (no data produced)")
    return data_files

def run_predictions(data_files, models, selected_models=None):
    for benchmark_name, csv_path in data_files.items():
        print(f"\nRunning predictions for benchmark: {benchmark_name}")
        csv_path = Path(csv_path)
        # Determine predictions output path
        if csv_path.stem.endswith("_w_predictions"):
            out_path = csv_path
            base_path = csv_path
        else:
            out_path = csv_path.with_name(f"{csv_path.stem}_w_predictions{csv_path.suffix}")
            base_path = csv_path

        # If predictions already exist, skip AI and use that file
        if out_path.exists():
            print(f"  Skipping model runs (found {out_path.name})")
            data_files[benchmark_name] = out_path
            continue

        if not base_path.exists():
            print(f"  Skipping {benchmark_name} (base file missing)")
            continue

        df = pd.read_csv(base_path)

        for model_name, predict_func in models.items():
            if selected_models and model_name not in selected_models:
                continue

            col_name = f"pred_{model_name}"
            if col_name in df.columns:
                print(f"  Skipping {model_name} (already exists)")
                continue

            print(f"  Running {model_name}...")
            predictions = []
            for idx, row in df.iterrows():
                if idx % 10 == 0 and idx > 0:
                    print(f"    Processed {idx} rows...")
                pred = predict_func(row)
                print(pred)
                predictions.append(pred)

            df[col_name] = predictions

    # Write predictions to a separate file and update mapping for this benchmark
    df.to_csv(out_path, index=False)
    data_files[benchmark_name] = out_path

def evaluate_predictions(data_files, evaluators):
    for benchmark_name, csv_path in data_files.items():
        if benchmark_name not in evaluators:
            print(f"No evaluator for {benchmark_name}, skipping evaluation")
            continue
        
        print(f"\nEvaluating predictions for: {benchmark_name}")
        if not Path(csv_path).exists():
            print(f"  Skipping {benchmark_name} (file missing)")
            continue
        df = pd.read_csv(csv_path)
        evaluators[benchmark_name](df, csv_path)

def generate_summary(data_files, output_path):
    summary_data = []
    
    for benchmark_name, csv_path in data_files.items():
        df = pd.read_csv(csv_path)
        pred_cols = [col for col in df.columns if col.startswith("pred_")]
        
        for pred_col in pred_cols:
            model_name = pred_col.replace("pred_", "")
            score_col = f"correct_{model_name}"
            
            if score_col in df.columns:
                accuracy = df[score_col].mean() * 100
                summary_data.append({
                    "Benchmark": benchmark_name,
                    "Model": model_name,
                    "Accuracy": f"{accuracy:.2f}%",
                    "Correct": df[score_col].sum(),
                    "Total": len(df)
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")
    print("\n" + summary_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="LLM Benchmarking Automation")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to run")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--skip-download", action="store_true", help="Skip benchmark download")
    parser.add_argument("--skip-predict", action="store_true", help="Skip predictions")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--summary-only", action="store_true", help="Only generate summary")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    benchmarks = load_benchmarks()
    models = load_models()
    evaluators = load_evaluators()
    
    print(f"Loaded {len(benchmarks)} benchmarks")
    print(f"Loaded {len(models)} models")
    print(f"Loaded {len(evaluators)} evaluators")
    
    if args.summary_only:
        data_files = {name: data_dir / f"{name}.csv" for name in benchmarks.keys()}
        data_files = {k: v for k, v in data_files.items() if v.exists()}
        generate_summary(data_files, data_dir / "summary.csv")
        return
    
    if not args.skip_download:
        data_files = prepare_benchmarks(benchmarks, data_dir, args.benchmarks)
    else:
        benchmark_names = args.benchmarks if args.benchmarks else benchmarks.keys()
        data_files = {name: data_dir / f"{name}.csv" for name in benchmark_names}
        data_files = {k: v for k, v in data_files.items() if v.exists()}
    
    if not args.skip_predict:
        run_predictions(data_files, models, args.models)
    
    if not args.skip_eval:
        evaluate_predictions(data_files, evaluators)
    
    generate_summary(data_files, data_dir / "summary.csv")

if __name__ == "__main__":
    main()
