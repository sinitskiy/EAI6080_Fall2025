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
        return 128
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


def load_benchmarks(selected: list[str] | None = None):
    benchmarks: dict[str, object] = {}
    benchmark_dir = Path(__file__).parent / "benchmarks"
    if not benchmark_dir.exists():
        return benchmarks

    allow = set([s.replace("-", "_") for s in selected]) if selected else None
    for file in benchmark_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        key = file.stem
        safe_name = key.replace("-", "_")
        if allow is not None and safe_name not in allow and key not in allow:
            continue
        module = _load_module_from_path(f"benchmarks.{safe_name}", file)
        if hasattr(module, "download_and_prepare"):
            benchmarks[key] = module.download_and_prepare
    return benchmarks

def load_models(selected: list[str] | None = None):
    models: dict[str, object] = {}
    model_dir = Path(__file__).parent / "models"
    if not model_dir.exists():
        return models

    allow = set([s.replace("-", "_") for s in selected]) if selected else None
    for file in model_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        key = file.stem
        safe_name = key.replace("-", "_")
        if allow is not None and safe_name not in allow and key not in allow:
            continue
        module = _load_module_from_path(f"models.{safe_name}", file)
        if hasattr(module, "predict"):
            models[key] = module.predict
    return models

def load_evaluators(selected_benchmarks: list[str] | None = None):
    evaluators: dict[str, object] = {}
    eval_dir = Path(__file__).parent / "evaluators"
    if not eval_dir.exists():
        return evaluators

    # Evaluators are keyed by benchmark stem; accept either exact stem or with _eval suffix
    allow = set([s.replace("-", "_") for s in selected_benchmarks]) if selected_benchmarks else None
    for file in eval_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        key = file.stem
        base = key[:-5] if key.endswith("_eval") else key
        safe_name = key.replace("-", "_")
        if allow is not None and base not in allow and key not in allow:
            continue
        module = _load_module_from_path(f"evaluators.{safe_name}", file)
        if hasattr(module, "evaluate"):
            # Store the module so we can access either evaluate or compute helpers
            evaluators[base] = module
    return evaluators

def _list_available_stems(kind: str) -> list[str]:
    folder = Path(__file__).parent / kind
    stems: list[str] = []
    if not folder.exists():
        return stems
    for file in folder.glob("*.py"):
        if not file.stem.startswith("_"):
            stems.append(file.stem)
    return stems

def _list_existing_benchmark_bases(data_dir: Path) -> list[str]:
    names: list[str] = []
    for f in data_dir.glob("*.csv"):
        n = f.stem
        if n.endswith("_pred") or "_pred_" in n or "_eval_" in n or n == "summary":
            continue
        names.append(n)
    return names

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

def run_predictions(data_files, models, selected_models=None, sample_n: int | None = None, seed: int | None = None):
    for benchmark_name, csv_path in data_files.items():
        print(f"\nRunning predictions for benchmark: {benchmark_name}")
        base_path = Path(csv_path)
        if not base_path.exists():
            print(f"  Skipping {benchmark_name} (base file missing)")
            continue

        # Write predictions to the dedicated predictions_and_evaluations folder
        pred_dir = Path(__file__).parent / "data" / "predictions_and_evaluations"
        pred_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(base_path)

        # Optional sampling for expensive LLMs
        subset_suffix = ""
        df_use = df
        if sample_n and sample_n > 0:
            n = min(sample_n, len(df))
            if n < len(df):
                rs = seed if seed is not None else 0
                df_use = df.sample(n=n, random_state=rs).reset_index(drop=True)
                subset_suffix = f"__sample{n}_seed{rs}"
                print(f"  Using a subset of {n}/{len(df)} rows (seed={rs})")

        for model_name, predict_func in models.items():
            if selected_models and model_name not in selected_models:
                continue

            pred_path = pred_dir / f"{benchmark_name}_pred_{model_name}{subset_suffix}.csv"
            if pred_path.exists():
                print(f"  Skipping {model_name} (file exists: {pred_path.name})")
                continue

            print(f"  Running {model_name}...")
            workers = _workers_for_model(model_name)
            preds = _predict_rows_parallel(df_use, predict_func, workers)
            out_df = pd.DataFrame({"id": df_use["id"], "answer": df_use.get("answer", pd.Series([None]*len(df_use))), "prediction": preds})
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

        pred_dir = Path(__file__).parent / "data" / "predictions_and_evaluations"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_files = list(pred_dir.glob(f"{benchmark_name}_pred_*.csv"))
        if not pred_files:
            print(f"  No prediction files found for {benchmark_name}")
            continue

        base_df = pd.read_csv(base_csv)
        ev_mod = evaluators[benchmark_name]

        for pf in pred_files:
            model_name = pf.stem.split("_pred_")[-1]
            pred_df = pd.read_csv(pf)
            merged = base_df.merge(pred_df, on="id", how="left")
            work_df = merged.copy()
            pred_col = f"pred_{model_name}"
            work_df[pred_col] = work_df["prediction"].fillna("")

            ok = None
            if hasattr(ev_mod, "compute_correct_series"):
                try:
                    ok = ev_mod.compute_correct_series(work_df, pred_col)
                except Exception as e:
                    print(f"  Evaluator error for {model_name}: {e}")
                    ok = None
            else:
                print(f"  Evaluator for {benchmark_name} lacks compute_correct_series, skipping {model_name}")
                continue

            if ok is None:
                continue

            corr_map = dict(zip(merged["id"], ok))
            pred_df["correct"] = pred_df["id"].map(corr_map).fillna(0).astype(int)
            pred_df.to_csv(pf, index=False)
            print(f"  Updated: {pf.name}")

def generate_summary(data_files, output_path):
    summary_data = []

    for benchmark_name, base_csv in data_files.items():
        base_csv = Path(base_csv)
        pred_dir = Path(__file__).parent / "data" / "predictions_and_evaluations"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_files = list(pred_dir.glob(f"{benchmark_name}_pred_*.csv"))
        if not pred_files:
            print(f"No prediction files found for {benchmark_name}, skipping in summary")
            continue

        for pf in pred_files:
            df = pd.read_csv(pf)
            if "correct" not in df.columns:
                continue
            model_name = pf.stem.split("_pred_")[-1]
            accuracy = df["correct"].mean() * 100
            summary_data.append({
                "Benchmark": benchmark_name,
                "Model": model_name,
                "Accuracy": f"{accuracy:.2f}%",
                "Correct": int(df["correct"].sum()),
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
    parser.add_argument("--sample", type=int, help="Random sample size per benchmark for predictions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    data_root = base_dir / "data"
    bench_data_dir = data_root / "benchmarks_data"
    pred_eval_dir = data_root / "predictions_and_evaluations"
    bench_data_dir.mkdir(parents=True, exist_ok=True)
    pred_eval_dir.mkdir(parents=True, exist_ok=True)

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
            base = bench_data_dir / f"{name}.csv"
            if base.exists():
                files[name] = base
        return files

    # Resolve selected benchmarks without loading modules upfront
    if args.benchmarks:
        selected_benchmarks = args.benchmarks
    else:
        # Prefer existing base CSVs; if none, list available scripts in benchmarks/
        bases = _list_existing_benchmark_bases(bench_data_dir)
        selected_benchmarks = bases if bases else _list_available_stems("benchmarks")

    data_files = {}
    if do_download:
        # Load only required benchmark modules
        benchmarks = load_benchmarks(selected_benchmarks)
        print(f"Loaded {len(benchmarks)} benchmarks")
        data_files = prepare_benchmarks(benchmarks, bench_data_dir, selected_benchmarks)
    else:
        data_files = existing_files_map(selected_benchmarks)

    if do_predict:
        # Ensure we have some input files (from download or existing)
        if not data_files:
            data_files = existing_files_map(selected_benchmarks)
        # Load only required model modules
        selected_models = args.models if args.models else _list_available_stems("models")
        models = load_models(selected_models)
        print(f"Loaded {len(models)} models")
        run_predictions(data_files, models, selected_models, args.sample, args.seed)

    if do_evaluate:
        # Load only required evaluator modules
        evaluators = load_evaluators(selected_benchmarks)
        print(f"Loaded {len(evaluators)} evaluators")
        evaluate_predictions(data_files, evaluators)

    if do_summary:
        data_files_for_summary = existing_files_map(selected_benchmarks)
        generate_summary(data_files_for_summary, pred_eval_dir / "summary.csv")

if __name__ == "__main__":
    main()
