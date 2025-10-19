# this evaluation is incorrect!!
import re

def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[\.,;:!\?\)\(\[\]\{\}\-\_\'\"]", "", s)
    return s

def compute_correct_series(df, pred_col: str):
    ok = []
    for _, r in df.iterrows():
        gt = _norm(r.get("answer", ""))
        pr = _norm(r.get(pred_col, ""))
        ok.append(1 if (gt and (pr.startswith(gt) or gt in pr)) else 0)
    return ok

def evaluate(df, csv_path):
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    for pc in pred_cols:
        model = pc.replace("pred_", "")
        cc = f"correct_{model}"
        if cc in df.columns:
            continue
        ok = compute_correct_series(df, pc)
        df[cc] = ok
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({
        "answer": ["E", "D"],
        "pred_gpt5nano": [
            "D. Both A and B Reason: Dilp2 is produced in IPCs and secreted into the hemolymph, which can cross the BBB to activate neural stem cells. Additionally, Dilp2 is transported to DRNs via direct innervation and retrograde transport, and DRN signaling contributes to NSC reactivation. Inhibiting either systemic Dilp2 or DRN function delays reactivation, and high insulin can drive reactivation even without feeding, indicating both sources (hemolymph-derived and DRN-associated Dilp2) can drive NSC reactivation.", 
            "C. Herring Reason: Fish like herrings have a highly sensitive lateral line system that detects very small water movements and vibrations. This could allow them to sense tiny disturbances in the water caused by a humanâ€™s muscle twitches, more so than the other animals listed."],
    })
    evaluate(df, "evaluation_output.csv")
    print(df)
    
# in both cases, we should get 0 (both answers are incorrect), but the current version of the code gives 1 in both cases