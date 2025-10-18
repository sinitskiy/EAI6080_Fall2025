import os

def predict(row):
    question = row.get("question", "")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(question)
        
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"
