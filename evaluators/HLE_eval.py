import re

def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[\.,;:!\?\)\(\[\]\{\}\-\_\'\"]", "", s)
    return s

def evaluate(df, csv_path):
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    for pc in pred_cols:
        model = pc.replace("pred_", "")
        cc = f"correct_{model}"
        if cc in df.columns:
            continue
        ok = []
        for _, r in df.iterrows():
            gt = _norm(r.get("answer", ""))
            pr = _norm(r.get(pc, ""))
            ok.append(1 if (gt and (pr.startswith(gt) or gt in pr)) else 0)
        df[cc] = ok
    df.to_csv(csv_path, index=False)
