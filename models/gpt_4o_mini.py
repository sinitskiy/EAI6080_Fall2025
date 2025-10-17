import os

def predict(row):
    question = row.get("question", "")
    
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"
