from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

# Downloading the most compressed version (Q4_0) to save space
MODEL_PATH = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_0.gguf" 
)

# CRITICAL SETTINGS FOR 512MB RAM:
llm = Llama(
    model_path=MODEL_PATH, 
    n_ctx=256,      # Reduced from 512 to 256 to save ~100MB RAM
    n_threads=1,    # Uses only 1 CPU core to keep memory overhead low
    use_mmap=False  # Disables memory mapping to stay within hard limits
)

@app.get("/")
def home():
    return {"status": "Model is alive on Render!"}

@app.get("/ask")
def ask(prompt: str):
    # Short response limit to prevent RAM spikes during generation
    output = llm(f"Q: {prompt} A: ", max_tokens=50, stop=["Q:"])
    return {"response": output["choices"][0]["text"]}
