"""
Gemini 2.5 Pro model for running predictions on benchmarks.
"""

import google.generativeai as genai
import pandas as pd
import time
import os
from typing import Optional

def predict(benchmark_df: pd.DataFrame, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Run predictions using Gemini-2.5-Pro on a benchmark dataset.
    
    Args:
        benchmark_df: DataFrame with columns ['id', 'question', 'image_path', 'answer']
        api_key: Google API key (if None, reads from GOOGLE_API_KEY env variable)
    
    Returns:
        DataFrame with columns ['id', 'answer', 'prediction']
    """
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as GOOGLE_API_KEY environment variable")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    results = []
    
    print(f"Processing {len(benchmark_df)} questions with Gemini-2.5-Pro...")
    
    for idx, row in benchmark_df.iterrows():
        question = row['question']
        ground_truth = row['answer']
        question_id = row['id']
        
        # Check if image is available
        image_path = row.get('image_path', None)
        has_image = image_path and pd.notna(image_path) and os.path.exists(image_path)
        
        try:
            if has_image:
                # For multimodal questions with images
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    response = model.generate_content([question, img])
                except Exception as img_error:
                    print(f"Warning: Could not load image for question {question_id}. Processing as text only.")
                    response = model.generate_content(question)
            else:
                # Text-only questions
                response = model.generate_content(question)
            
            prediction = response.text.strip()
            
            print(f"Question {idx+1}/{len(benchmark_df)}: ID={question_id}, Predicted")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error on question {question_id}: {error_msg[:150]}")
            
            # Handle quota limits
            if "quota" in error_msg.lower() or "429" in error_msg:
                print(f"\nQuota limit reached at question {idx+1}/{len(benchmark_df)}")
                print(f"Saving partial results...")
                # Return partial results
                if results:
                    return pd.DataFrame(results)
                else:
                    raise Exception("Quota exceeded before any predictions completed")
            
            prediction = "ERROR"
            
            # Wait before retrying
            time.sleep(5)
        
        results.append({
            'id': question_id,
            'answer': ground_truth,
            'prediction': prediction
        })
        
        # Rate limiting to avoid quota issues (free tier: 2 requests/minute)
        time.sleep(2)
    
    print(f"\nCompleted {len(results)} predictions")
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the model
    print("Testing Gemini-2.5-Pro model...")
    
    # Create sample data
    test_data = pd.DataFrame({
        'id': ['test_1', 'test_2'],
        'question': [
            'What is the capital of France?',
            'What is 2 + 2?'
        ],
        'image_path': [None, None],
        'answer': ['Paris', '4']
    })
    
    # Run predictions
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
    else:
        results = predict(test_data, api_key=api_key)
        print("\nTest Results:")
        print(results)
