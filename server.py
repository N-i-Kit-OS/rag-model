"""
server.py — HTTP API для RAG-чатбота.
Запуск: python server.py

Эндпоинты:
  POST /ask        {"question": "..."}          → {"answer": "...", "sources": [...]}
  GET  /health                                    → {"status": "ok", "documents": N}
"""

import logging
import time

import chromadb
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from config import (
    CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, LLM_MODEL, LLM_URL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    HF_TOKEN, SYSTEM_PROMPT,
    TOP_K, SCORE_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Medical RAG Chatbot")

db = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = db.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
embed_client = InferenceClient(model=EMBEDDING_MODEL, token=HF_TOKEN)

log.info(f"ChromaDB: {collection.count()} документов")


class AskRequest(BaseModel):
    question: str



@app.get("/health")
def health():
    return {"status": "ok", "documents": collection.count()}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "Пустой вопрос"}, 400)

    try:
        q_emb = embed_client.feature_extraction(question)
        if hasattr(q_emb, "tolist"):
            q_emb = q_emb.tolist()

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        sources = []
        context_parts = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                sim = round(1 - dist, 4)
                if sim >= SCORE_THRESHOLD:
                    sources.append({**meta, "similarity": sim})
                    context_parts.append(
                        f"Тема: {meta['topic']}\n"
                        f"Вопрос: {meta['question']}\n"
                        f"Ответ: {meta['answer']}"
                    )

        context = "\n\n---\n\n".join(context_parts) if context_parts else "Релевантных данных нет."

        user_prompt = (
            f"Контекст из базы медицинских знаний:\n---\n{context}\n---\n\n"
            f"Вопрос пользователя: {question}\n\n"
            f"Ответь на вопрос, опираясь на контекст. "
            f"Если релевантных данных нет — так и скажи и порекомендуй обратиться к врачу."
        )

        llm_resp = requests.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
            timeout=120,
        )

        if llm_resp.status_code != 200:
            log.error(f"LLM ошибка {llm_resp.status_code}: {llm_resp.text[:300]}")
            return JSONResponse(
                {"error": f"LLM API ошибка {llm_resp.status_code}: {llm_resp.text[:200]}"},
                502,
            )

        answer = llm_resp.json()["choices"][0]["message"]["content"].strip()

        return {
            "answer": answer,
            "sources": sources,
        }

    except requests.exceptions.Timeout:
        return JSONResponse({"error": "Таймаут LLM API"}, 504)
    except Exception as e:
        log.exception("Ошибка")
        return JSONResponse({"error": str(e)}, 500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)