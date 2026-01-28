from fastapi import FastAPI
from fastapi.responses import HTMLResponse # Add this
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

# SmolLM2-135M: The safest choice for Render's 512MB RAM
MODEL_PATH = hf_hub_download(
    repo_id="bartowski/SmolLM2-135M-Instruct-GGUF",
    filename="SmolLM2-135M-Instruct-Q4_K_M.gguf"
)

llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=1, use_mmap=False)

# NEW: This serves the chat interface when you visit your URL
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/ask")
def ask(prompt: str):
    output = llm(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", 
        max_tokens=100, 
        stop=["<|im_end|>"]
    )
    return {"response": output["choices"][0]["text"]}
