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
    
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": q}],
            # temperature: for this model, only the default (1) value is supported
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    test_row = {"question": "How many letters R are in strawberry?"}
    print(predict(test_row))
    