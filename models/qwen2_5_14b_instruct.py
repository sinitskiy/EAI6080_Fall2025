import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def predict(questions, model_path="/courses/EAI6080.202615/students/lakhankiya.t/models/qwen2.5-14b-instruct", max_new_tokens=128):
    """
    Input:  list of dicts [{'id':..., 'question':..., 'answer':...}]
    Output: list of dicts [{'id':..., 'answer':..., 'prediction':...}]
    """

    print(f"🔗 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # ✅ Debug: confirm how many questions were received
    print(f"🧪 predict() called with {len(questions)} questions")
    if len(questions) == 0:
        print("⚠️ No questions received by model — dataset may not have been passed correctly.")
        return []

    results = []
    total = len(questions)
    print(f"📘 Generating predictions for {total} questions...")

    for i, item in enumerate(questions, 1):
        qid = item.get("id", i)
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Detect MCQ vs open-form
        if any(x in question for x in ["A.", "A)", "A:"]):
            prompt = f"Question:\n{question}\n\nChoose the correct answer (A, B, C, or D) and explain.\nAnswer:"
        else:
            prompt = f"Question:\n{question}\n\nAnswer the question and explain your reasoning.\nAnswer:"

        try:
            output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=True)[0]["generated_text"]
            if "Answer:" in output:
                pred = output.split("Answer:", 1)[-1].strip()
            else:
                pred = output.strip()

            prediction = f"Answer: {pred}\nReasoning: (model reasoning follows if present)"
        except Exception as e:
            print(f"⚠️ Error on Q{i}: {e}")
            prediction = "Error"

        results.append({"id": qid, "answer": answer, "prediction": prediction})

        if i % 5 == 0 or i == total:
            print(f"✅ Processed {i}/{total}")

    # ✅ Debug confirmation before returning
    print(f"💾 Returning {len(results)} predictions to main.py")
    print("✅ Prediction complete!")
    return results
