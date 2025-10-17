def predict(row):
    question = row.get("question", "")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"
