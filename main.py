from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

# Download a tiny 4-bit quantized model (approx 350MB)
# The original Qwen repo uses dots (qwen2.5) and underscores differently
MODEL_PATH = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"  # Note the 2.5 and the dots
)

# Load model into RAM (limited context to save memory)
llm = Llama(model_path=MODEL_PATH, n_ctx=512)

@app.get("/")
def home():
    return {"status": "Model is running on Render!"}

@app.get("/ask")
def ask(prompt: str):
    output = llm(f"Q: {prompt} A: ", max_tokens=100, stop=["Q:", "\n"])
    return {"response": output["choices"][0]["text"]}
