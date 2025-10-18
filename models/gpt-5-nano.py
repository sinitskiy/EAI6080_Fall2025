import os
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def predict(row):
    q = row.get("question", "")
    if q == "":
        return "ERROR: The question provided is empty."
    # if this ML model can take an image as input, get it here
    
    try:
        client = _get_client()
        params = {
            "model": "gpt-5-nano",
            "input": q,
            # remember to add image for ML models that can work with them,
        }
        resp = client.responses.create(**params)
        output = getattr(resp, "output_text", "").strip()
        if output != "":
            return output
        else:
            # include minimal diagnostics to understand why it's empty
            finish = getattr(resp, "finish_reasons", None)
            usage = getattr(resp, "usage", None)
            try:
                out_tok = getattr(usage, "output_tokens", None)
            except Exception:
                out_tok = None
            return f"ERROR: Empty response (finish_reasons={finish}, output_tokens={out_tok})"
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    row = {"question": "How many letters R are in strawberry?"}
    print(predict(row))