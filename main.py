import argparse
import importlib
import importlib.util
# from importlib.machinery import SourceFileLoader
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import local_secrets

def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    raise ImportError(f"Cannot load module {module_name} from {file_path}")


def _workers_for_model(model_name: str) -> int:
    name = (model_name or "").lower()
    if name.startswith("gpt"):
        return 32
    return 1


def _predict_rows_parallel(df: pd.DataFrame, predict_func, workers: int):
    if workers <= 1 or len(df) <= 1:
        return [predict_func(row) for _, row in df.iterrows()]

    results = [None] * len(df)

    def _task(i: int):
        row = df.iloc[i]
        return i, predict_func(row)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, i) for i in range(len(df))]
        for fut in as_completed(futures):
            i, pred = fut.result()
            results[i] = pred

    return results


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
        base_path = Path(csv_path)
        if not base_path.exists():
            print(f"  Skipping {benchmark_name} (base file missing)")
            continue

        df = pd.read_csv(base_path)

        for model_name, predict_func in models.items():
            if selected_models and model_name not in selected_models:
                continue

            pred_path = base_path.parent / f"{benchmark_name}_pred_{model_name}.csv"
            if pred_path.exists():
                print(f"  Skipping {model_name} (file exists: {pred_path.name})")
                continue

            print(f"  Running {model_name}...")
            workers = _workers_for_model(model_name)
            preds = _predict_rows_parallel(df, predict_func, workers)
            out_df = pd.DataFrame({
                "id": df["id"],
                "prediction": preds,
            })
            out_df.to_csv(pred_path, index=False)
            print(f"  Saved: {pred_path.name}")

def evaluate_predictions(data_files, evaluators):
    for benchmark_name, base_csv in data_files.items():
        if benchmark_name not in evaluators:
            print(f"No evaluator for {benchmark_name}, skipping evaluation")
            continue

        print(f"\nEvaluating predictions for: {benchmark_name}")
        base_csv = Path(base_csv)
        if not base_csv.exists():
            print(f"  Skipping {benchmark_name} (base file missing)")
            continue

        pred_files = list(base_csv.parent.glob(f"{benchmark_name}_pred_*.csv"))
        if not pred_files:
            print(f"  No prediction files found for {benchmark_name}")
            continue

        base_df = pd.read_csv(base_csv)
        ev = evaluators[benchmark_name]

        for pf in pred_files:
            model_name = pf.stem.split("_pred_")[-1]
            pred_df = pd.read_csv(pf)
            merged = base_df.merge(pred_df, on="id", how="left")
            work_df = merged.copy()
            pred_col = f"pred_{model_name}"
            work_df[pred_col] = work_df["prediction"].fillna("")

            out_eval = base_csv.parent / f"{benchmark_name}_eval_{model_name}.csv"
            ev(work_df, out_eval)
            print(f"  Wrote eval: {out_eval.name}")

def generate_summary(data_files, output_path):
    summary_data = []

    for benchmark_name, base_csv in data_files.items():
        base_csv = Path(base_csv)
        eval_files = list(base_csv.parent.glob(f"{benchmark_name}_eval_*.csv"))
        if not eval_files:
            print(f"No eval files found for {benchmark_name}, skipping in summary")
            continue

        for ef in eval_files:
            df = pd.read_csv(ef)
            corr_cols = [c for c in df.columns if c.startswith("correct_")]
            if not corr_cols:
                continue
            score_col = corr_cols[0]
            model_name = score_col.replace("correct_", "")
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
    if not summary_df.empty:
        print("\n" + summary_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="LLM Benchmarking Automation")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to run")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--all", action="store_true", help="Run download, predict, evaluate, and summary")
    parser.add_argument("--download", action="store_true", help="Download/prepare benchmarks")
    parser.add_argument("--predict", action="store_true", help="Run model predictions")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions")
    parser.add_argument("--summary", action="store_true", help="Generate summary table")
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

    # Determine which phases to run
    run_all = args.all or not any([args.download, args.predict, args.evaluate, args.summary])
    do_download = run_all or args.download
    do_predict = run_all or args.predict
    do_evaluate = run_all or args.evaluate
    do_summary = run_all or args.summary

    # Helper: build mapping of existing files for selected benchmarks
    def existing_files_map(selected_names):
        files = {}
        for name in selected_names:
            base = data_dir / f"{name}.csv"
            if base.exists():
                files[name] = base
        return files

    selected_benchmarks = args.benchmarks if args.benchmarks else list(benchmarks.keys())

    data_files = {}
    if do_download:
        data_files = prepare_benchmarks(benchmarks, data_dir, selected_benchmarks)
    else:
        data_files = existing_files_map(selected_benchmarks)

    if do_predict:
        # Ensure we have some input files (from download or existing)
        if not data_files:
            data_files = existing_files_map(selected_benchmarks)
        run_predictions(data_files, models, args.models)

    if do_evaluate:
        evaluate_predictions(data_files, evaluators)

    if do_summary:
        data_files_for_summary = existing_files_map(selected_benchmarks)
        generate_summary(data_files_for_summary, data_dir / "summary.csv")

if __name__ == "__main__":
    main()
