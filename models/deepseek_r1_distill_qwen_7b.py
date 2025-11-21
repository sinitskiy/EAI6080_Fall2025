"""
DeepSeek-R1-Distill-Qwen-7B model for benchmark predictions.
Compatible with: python main.py --predict --models deepseek_r1_distill_qwen_7b
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_model = None
_tokenizer = None
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def _get_model():
    """Load model and tokenizer once."""
    global _model, _tokenizer
    if _model and _tokenizer:
        return _model, _tokenizer

    print(f"🔄 Loading model: {MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, device_map="auto"
        )
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32, device_map="auto"
        )
        print("⚠️ Running on CPU.")
    return _model, _tokenizer


def _build_prompt(question: str, answer_type: str) -> str:
    """Format the prompt based on question type."""
    if (answer_type or "").lower() == "multiplechoice":
        return (
            "You are a biomedical reasoning assistant. "
            "Choose the correct option.\n\n"
            f"Question:\n{question}\n\n"
            "Answer with only the letter: A, B, C, or D."
        )
    return (
        "You are a biomedical reasoning assistant. "
        "Provide a concise, factual answer.\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def _extract_choice(text: str) -> str:
    """Extract A–D letter if present."""
    if not isinstance(text, str):
        return str(text)
    for ch in "ABCDabcd":
        if ch in text:
            return ch.upper()
    return text.strip()


def predict(row):
    """Return model prediction for one row."""
    q = str(row.get("question", "")).strip()
    if not q:
        return "ERROR: Empty question."

    answer_type = str(row.get("answer_type", "open")).strip()
    try:
        model, tokenizer = _get_model()
        prompt = _build_prompt(q, answer_type)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64 if answer_type.lower() != "multiplechoice" else 16,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return _extract_choice(text) if answer_type.lower() == "multiplechoice" else text
    except Exception as e:
        return f"ERROR: {e}"
