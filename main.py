from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

# SmolLM2-135M is significantly smaller and will fit easily in 512MB RAM
MODEL_PATH = hf_hub_download(
    repo_id="bartowski/SmolLM2-135M-Instruct-GGUF",
    filename="SmolLM2-135M-Instruct-Q4_K_M.gguf"
)

# Load with extreme memory efficiency
llm = Llama(
    model_path=MODEL_PATH, 
    n_ctx=512,       # Standard context size
    n_threads=1,     # 1 thread prevents CPU-related memory spikes
    use_mmap=False   # Essential for low-memory environments
)

@app.get("/")
def home():
    return {"status": "SmolLM is active and stable!"}

@app.get("/ask")
def ask(prompt: str):
    output = llm(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", 
        max_tokens=100, 
        stop=["<|im_end|>"]
    )
    return {"response": output["choices"][0]["text"]}
