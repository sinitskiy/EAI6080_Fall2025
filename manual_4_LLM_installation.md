# LLM Installation & Usage Manual

## Overview
This manual covers installation and usage for four LLMs across three platforms:
- **Windows Laptop**
- **Mac Laptop** 
- **Northeastern University HPC Cluster**

**Models Covered:**
- DeepSeek-R1-Distill-Qwen-7B
- Qwen/Qwen2.5-14B-Instruct
- GPT-5-mini (API)
- Gemini-2.5-pro (API)

---

## Part 1: API-Based Models (All Platforms)

### GPT-5-mini (OpenAI API)

**Setup:**
1. Register at [platform.openai.com](https://platform.openai.com)
2. Generate API key from account settings
3. Install library:
```bash
pip install openai
```

**Usage:**
```python
import openai

client = openai.OpenAI(api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Your prompt here"}]
)
output = response.choices[0].message.content
print(output)
```

**Documentation:** [OpenAI GPT-5-mini Docs](https://platform.openai.com/docs/models/gpt-5-mini)

---

### Gemini-2.5-pro (Google AI API)

**Setup:**
1. Register at [ai.google.dev](https://ai.google.dev)
2. Generate API key from Google AI Studio
3. Install library:
```bash
pip install google-generativeai
```

**Usage:**
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.5-pro")
response = model.generate_content("Your prompt here")
output = response.text
print(output)
```

**Documentation:** [Gemini API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)

---

## Part 2: Local Model Installation

### DeepSeek-R1-Distill-Qwen-7B

#### Windows Installation

**Prerequisites:**
- Python 3.10 or higher
- 16GB+ RAM recommended
- GPU optional but recommended

**Method 1: Using Transformers**
```bash
# Create virtual environment
python -m venv deepseek-env
deepseek-env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece
```

**Usage:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain quantum computing briefly."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Method 2: Using LM Studio (Easier)**
1. Download [LM Studio](https://lmstudio.ai/)
2. Open model browser
3. Search for "DeepSeek R1 Distill"
4. Download and run via GUI interface

---

#### Mac Installation

**Prerequisites:**
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- 16GB+ RAM

**Setup:**
```bash
# Create conda environment
conda create -n deepseek python=3.10 -y
conda activate deepseek

# Install dependencies
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install transformers accelerate sentencepiece

# Enable MPS fallback for Metal GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Usage:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="mps"  # Use Metal Performance Shaders
)

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

#### HPC Cluster Installation (Northeastern University)

**Step 1: Access Cluster**
```bash
# SSH access
ssh username@login.discovery.neu.edu

# OR use web portal
# https://ood.explorer.northeastern.edu/
```

**Step 2: Setup Environment**
```bash
# Load CUDA module
module load cuda/12.1

# Create conda environment
conda create -n deepseek_env python=3.10 -y
conda activate deepseek_env

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece huggingface_hub
```

**Step 3: Download Model**
```bash
# Login to Hugging Face (if gated model)
huggingface-cli login

# Download model to scratch directory
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', \
cache_dir='/scratch/$USER/models'); \
AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', \
cache_dir='/scratch/$USER/models')"
```

**Step 4: Create Run Script (run_deepseek.py)**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
cache_dir = "/scratch/xue.re/models"  # Replace with your username

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = " ".join(sys.argv[1:]) or "Explain machine learning."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Step 5: Request GPU and Run**
```bash
# Interactive session
srun --partition=gpu --gres=gpu:1 --mem=32G --time=01:00:00 --pty /bin/bash

# Activate environment
conda activate deepseek_env

# Run model
python run_deepseek.py "Your prompt here"
```

**Step 6: SLURM Batch Job (Optional)**
```bash
#!/bin/bash
#SBATCH --job-name=deepseek_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/deepseek_output.txt

module load cuda/12.1
source activate deepseek_env

python run_deepseek.py "Explain deep learning in simple terms."
```

Submit with: `sbatch job_script.sh`

---

### Qwen/Qwen2.5-14B-Instruct

#### Windows Installation

**Prerequisites:**
- Python 3.10+
- 32GB+ RAM recommended (14B parameter model)
- GPU with 24GB+ VRAM recommended

**Setup:**
```bash
# Create environment
python -m venv qwen-env
qwen-env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece pillow
```

**Usage:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is artificial intelligence?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```

---

#### Mac Installation

**Prerequisites:**
- Apple Silicon (M1/M2/M3/M4)
- 32GB+ unified memory recommended

**Setup:**
```bash
# Create environment
conda create -n qwen python=3.10 -y
conda activate qwen

# Install dependencies
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install transformers accelerate sentencepiece pillow

# Enable MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Usage:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="mps"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain neural networks."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("mps")

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```

---

#### HPC Cluster Installation

**Setup:**
```bash
# Access cluster
ssh username@login.discovery.neu.edu

# Load modules
module load cuda/12.1

# Create environment
conda create -n qwen_env python=3.10 -y
conda activate qwen_env

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece pillow huggingface_hub
```

**Create run_qwen.py:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

model_name = "Qwen/Qwen2.5-14B-Instruct"
cache_dir = "/scratch/xue.re/models"  # Update username

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

user_prompt = " ".join(sys.argv[1:]) or "What is deep learning?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```

**Run with GPU:**
```bash
# Interactive session
srun --partition=gpu --gres=gpu:a100:1 --mem=64G --time=02:00:00 --pty /bin/bash

conda activate qwen_env
python run_qwen.py "Your prompt here"
```

**SLURM Script:**
```bash
#!/bin/bash
#SBATCH --job-name=qwen_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/qwen_output.txt

module load cuda/12.1
source activate qwen_env

python run_qwen.py "Explain reinforcement learning."
```

---

## Common Issues & Solutions

### All Platforms

**Issue: Out of Memory (OOM)**
- Solution: Use smaller batch size, enable gradient checkpointing, or use quantization:
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Issue: Slow Download**
- Solution: Use `huggingface-cli` with resume capability:
```bash
huggingface-cli download model_name --resume-download
```

**Issue: CUDA Not Available**
- Windows/Linux: Verify CUDA installation with `nvidia-smi`
- Mac: MPS should be available on M1+ chips
- Check with: `python -c "import torch; print(torch.cuda.is_available())"`

### HPC-Specific Issues

**Issue: Permission Denied**
- Solution: Use `/scratch/username/` instead of `/home/username/` for large files

**Issue: Module Not Found**
- Solution: Ensure environment is activated: `conda activate env_name`

**Issue: GPU Not Available in Login Node**
- Solution: Always request GPU via `srun` or SLURM job

**Issue: Disk Quota Exceeded**
- Solution: Clean cache or use scratch directory:
```bash
export HF_HOME=/scratch/$USER/huggingface_cache
```

---

## Performance Tips

### Memory Optimization
1. Use `torch.float16` or `torch.bfloat16` instead of `float32`
2. Enable gradient checkpointing for training
3. Use 4-bit or 8-bit quantization for inference
4. Clear CUDA cache: `torch.cuda.empty_cache()`

### Speed Optimization
1. Use `device_map="auto"` for automatic GPU allocation
2. Increase `max_new_tokens` batch processing
3. Use Flash Attention 2 if available
4. Pre-download models to avoid runtime delays

### Storage Management
- Models range from 7GB (DeepSeek-7B) to 28GB (Qwen-14B)
- Use symbolic links to share models across environments
- Set custom cache: `export HF_HOME=/path/to/cache`

---

## Quick Reference

| Model | Size | RAM Required | GPU VRAM | Best Platform |
|-------|------|--------------|----------|---------------|
| DeepSeek-R1-Distill-Qwen-7B | ~14GB | 16GB+ | 16GB+ | All platforms |
| Qwen2.5-14B-Instruct | ~28GB | 32GB+ | 24GB+ | HPC, High-end Mac |
| GPT-5-mini (API) | N/A | N/A | N/A | All (internet required) |
| Gemini-2.5-pro (API) | N/A | N/A | N/A | All (internet required) |

**Recommended GPU Specifications:**
- Windows/Linux: NVIDIA RTX 3090/4090 or A100
- Mac: M2 Max/Ultra or M3 Max with 32GB+ unified memory
- HPC: A100 (40GB or 80GB)
