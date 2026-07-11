import os
from pathlib import Path

# path
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
CSV_PATH = Path(os.environ.get("CSV_PATH", BASE_DIR / "medical_dialogues_clean.csv"))

# hf api
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# emb
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# llm
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_URL = "https://router.huggingface.co/v1/chat/completions"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# name of collection
COLLECTION_NAME = "medical_dialogues"

# rag
TOP_K = 5
SCORE_THRESHOLD = 0.3

# prompt
SYSTEM_PROMPT = (
    "Ты — медицинский информационный ассистент. "
    "Отвечай на русском, опираясь ТОЛЬКО на контекст. "
    "Не ставь диагноз — рекомендуй обратиться к врачу. "
    "Давай развёрнутые, понятные ответы."
)