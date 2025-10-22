import google.generativeai as genai
import pandas as pd
import time
import os
from pathlib import Path

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as GOOGLE_API_KEY environment variable")
        genai.configure(api_key=api_key)
        _client = genai.GenerativeModel('gemini-2.5-pro')
    return _client

def predict(row):
    question = row.get("question", "")
    if question == "":
        return "ERROR: The question provided is empty."
    
    # Check if image is available
    image_path = row.get('image_path', None)
    # Guard against None/NaN and invalid paths
    has_image = False
    if image_path and pd.notna(image_path):
        try:
            ipath = Path(__file__).parent / '..' / str(image_path)
            has_image = ipath.exists()
        except Exception:
            has_image = False
    
    prediction = ""
    try:
        client = _get_client()
        if has_image:
            # For multimodal questions with images
            try:
                from PIL import Image
                img = Image.open(ipath)
                response = client.generate_content([question, img])
            except Exception as img_error:
                print(f"Warning: Could not load image for question {question}. Processing as text only.")
                response = client.generate_content(question)
        else:
            # Text-only questions
            response = client.generate_content(question)
        if hasattr(response, 'text') and isinstance(response.text, str):
            prediction = response.text.strip()
        else:
            prediction = ""
    except Exception as e:
        error_msg = str(e)
        print(f"Error on question {question}: {error_msg[:150]}")
        prediction = ""
    
    # Rate limiting to avoid quota issues (free tier: 2 requests/minute)
    # time.sleep(2)
    
    return prediction


if __name__ == "__main__":
    row = {"question": "What is in this picture?", 'image_path': "data\\benchmarks_data\\HLE_images\\1500.png"}
    print(predict(row))