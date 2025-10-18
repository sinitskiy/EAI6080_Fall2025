
import os
import re
import time
import gc
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

OPENAI_API_KEY = ""

MODEL = "gpt-5-mini"

DATA_PATHS = {
    "HLE": r"Your Path\test-00000-of-00001.parquet",
}

OUTPUT_DIR = r"Your Path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 50
MAX_RETRIES = 2
BACKOFF_BASE = 2.0
MAX_INPUT_CHARS = 4000

client = OpenAI(api_key=OPENAI_API_KEY)

def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:\-–—!?\u3002\uFF0C]", "", s)
    return s

def find_col(cols, keys):
    for k in keys:
        for c in cols:
            if k.lower() in str(c).lower():
                return c
    return None

def load_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unknown file format: {path}")

    q_col = find_col(df.columns, ["question","prompt","query","text","input"])
    a_col = find_col(df.columns, ["answer","final_answer","target","label"])
    # Keep using whatever 'subject-like' column exists as the subject source.
    s_col = find_col(df.columns, ["subject","domain","category","topic"])
    c_col = find_col(df.columns, ["choices","options"])

    if q_col is None:
        raise RuntimeError(f"No question column found in {path}")
    df = df.rename(columns={q_col:"question"})
    if a_col: df = df.rename(columns={a_col:"answer"})
    if s_col: df["subject"] = df[s_col]
    else: df["subject"] = "unknown"
    if c_col: df["choices"] = df[c_col]
    else: df["choices"] = None

    return df[["question","answer","subject","choices"]].dropna(subset=["question"]).reset_index(drop=True)

CATEGORY_RULES = [
    ("Engineering", [
        "engineering","electrical","mechanical","civil","aerospace","materials",
        "industrial","chemical engineering","control","signal processing","communications",
        "power systems","robotics","mechatronics","structural","manufacturing"
    ]),
    ("Math", [
        "math","algebra","calculus","geometry","trigonometry","probability",
        "statistics","number theory","combinatorics","set theory","logic","arithmetic",
        "topology","analysis","linear algebra","discrete math","differential equations"
    ]),
    ("Computer Science/Artificial Intelligence", [
        "computer science","artificial intelligence","ai","machine learning","ml","deep learning","dl",
        "algorithm","algorithms","data structure","data structures","programming","coding","software",
        "operating system","database","databases","network","networks","compiler","compilers",
        "nlp","natural language processing","computer vision","cybersecurity","computing","information retrieval"
    ]),
    ("Physics", [
        "physics","mechanics","electromagnetism","thermodynamics","optics","quantum",
        "relativity","astrophysics","astronomy","nuclear","solid state","condensed matter"
    ]),
    ("Chemistry", [
        "chemistry","organic","inorganic","physical chemistry","analytical chemistry","stoichiometry",
        "spectroscopy","thermochemistry","electrochemistry"
    ]),
    ("Biology/Medicine", [
        "biology","medicine","medical","anatomy","physiology","genetics","microbiology","immunology",
        "pharmacology","neuroscience","pathology","zoology","botany","ecology","biomedical",
        "biochemistry","cell biology","molecular biology","virology","oncology","endocrinology"
    ]),
    ("Humanities/Social Science", [
        "history","philosophy","linguistics","psychology","sociology","economics","political science",
        "law","anthropology","ethics","geography","education","art","arts","literature","music",
        "humanities","social science","archaeology","communication studies","cognitive science"
    ]),
    ("Other", ["other"]),
]

def subject_to_category(subject: str) -> str:
    s = norm(subject)
    for cat, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in s:
                return cat
    return "Other"

def make_prompt(q, choices=None):
    if isinstance(choices, (list,tuple)) and len(choices) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opts = "\n".join([f"{letters[i]}. {c}" for i,c in enumerate(choices[:len(letters)])])
        return f"You are an expert. Answer with ONLY one letter (A,B,C,...).\n\nQuestion:\n{q}\n\nChoices:\n{opts}\n\nAnswer:"
    return f"You are an expert. Provide ONLY the final short answer (no explanation).\n\nQuestion:\n{q}\n\nAnswer:"

def extract_letter(ans: str):
    m = re.search(r"\b([A-E])\b", ans or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""

def ask_gpt(prompt: str):
    messages = [
        {"role": "system", "content": "You are a precise evaluator. Answer succinctly."},
        {"role": "user", "content": prompt}
    ]
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages)
            return resp.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError, APIStatusError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            return f"[Error: {e}]"
        except Exception as e:
            return f"[Error: {e}]"
    return f"[Error: {last_err}]" if last_err else "[Error: Unknown]"

def eval_dataset(name: str, df: pd.DataFrame):
    out_path = os.path.join(OUTPUT_DIR, f"{name}_gpt5mini.csv")
    done = 0
    if os.path.exists(out_path):
        try:
            done = len(pd.read_csv(out_path))
        except Exception:
            done = 0

    n = len(df)
    for beg in range(done, n, CHUNK_SIZE):
        end = min(beg + CHUNK_SIZE, n)
        chunk = df.iloc[beg:end].copy()
        preds, corrects = [], []

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"{name}[{beg}:{end})"):
            q = row["question"]
            a = row.get("answer", "")
            subj = row.get("subject", "unknown")
            choices = row["choices"] if isinstance(row["choices"], (list,tuple)) else None
            prompt = make_prompt(q, choices)
            if len(prompt) > MAX_INPUT_CHARS:
                prompt = prompt[:MAX_INPUT_CHARS] + "\n...[truncated]"

            ans = ask_gpt(prompt)
            if ans.startswith("[Error:"):
                pred = ""
            elif choices:
                pred = extract_letter(ans)
            else:
                pred = ans.split("\n")[0].strip()

            score = 1.0 if norm(pred) == norm(a) else 0.0
            preds.append(pred)
            corrects.append(score)

        chunk["prediction"] = preds
        chunk["correct"] = corrects
        chunk["category"] = chunk["subject"].apply(subject_to_category)

        save_cols = [c for c in ["question","answer","subject","category","prediction","correct"] if c in chunk.columns]
        chunk_to_save = chunk[save_cols]

        chunk_to_save.to_csv(out_path, mode="a" if beg else "w", index=False, header=not beg)
        gc.collect()

    full = pd.read_csv(out_path)
    acc = full["correct"].mean() if len(full) else 0.0
    per_sub = full.groupby("subject")["correct"].mean().sort_values(ascending=False)
    per_cat = full.groupby("category")["correct"].mean().sort_values(ascending=False)

    per_sub.to_csv(os.path.join(OUTPUT_DIR, f"{name}_per_subject.csv"))
    per_cat.to_csv(os.path.join(OUTPUT_DIR, f"{name}_per_category.csv"))
    return acc, per_sub, per_cat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Choose one: HLE | HLE-Gold-Bio-Chem | BixBench")
    args = parser.parse_args()
    name = args.dataset.strip()

    if name not in DATA_PATHS:
        raise SystemExit(f"❌ Unknown dataset: {name}\nAvailable: {list(DATA_PATHS.keys())}")

    path = DATA_PATHS[name]
    if not os.path.exists(path):
        raise SystemExit(f"❌ File not found: {path}")

    print(f"\n===== Running {name} =====")
    df = load_any(path)
    acc, per_sub, per_cat = eval_dataset(name, df)
    print(f"[RESULT] {name}: ACC={acc*100:.2f}%")
    print("[OK] Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
