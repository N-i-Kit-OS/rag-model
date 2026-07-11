import csv
import logging
import time
from pathlib import Path

import chromadb
from huggingface_hub import InferenceClient

from config import (
    CSV_PATH, CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, HF_TOKEN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_csv():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            q = row.get("user_question_clean", "").strip()
            a = row.get("assistant_answer_clean", "").strip()
            topic = row.get("topic", "").strip()
            doctor = row.get("to_doctor", "").strip()
            if q and a:
                rows.append({
                    "id": str(idx),
                    "text": f"Тема: {topic}\nВопрос: {q}\nОтвет: {a}",
                    "topic": topic,
                    "question": q,
                    "answer": a,
                    "doctor": doctor,
                })
    log.info(f"Загружено {len(rows)} диалогов")
    return rows


def index():
    if not HF_TOKEN:
        raise ValueError("Установите HF_TOKEN: export HF_TOKEN=hf_...")

    rows = load_csv()
    if not rows:
        return

    client = InferenceClient(model=EMBEDDING_MODEL, token=HF_TOKEN)
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        db.delete_collection(COLLECTION_NAME)
        log.info("Старая коллекция удалена")
    except Exception:
        pass

    collection = db.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(rows)
    for i, row in enumerate(rows):
        emb = client.feature_extraction(row["text"])
        if hasattr(emb, "tolist"):
            emb = emb.tolist()

        collection.add(
            ids=[row["id"]],
            embeddings=[emb],
            documents=[row["text"]],
            metadatas=[{
                "topic": row["topic"],
                "question": row["question"],
                "answer": row["answer"],
                "doctor": row["doctor"],
            }],
        )

        if (i + 1) % 50 == 0 or (i + 1) == total:
            log.info(f"Проиндексировано {i + 1}/{total}")

        if i < total - 1:
            time.sleep(0.3)

    log.info(f"Готово! В коллекции {collection.count()} документов")


if __name__ == "__main__":
    index()