# RAG Chatbot

RAG-чатбот для медицинской информационной системы на русском языке.  
Ищет релевантные диалоги в векторной базе через косинусное сходство и генерирует ответ с помощью LLM.

### Как работает RAG

1. **Индексация** (один раз): каждый диалог из CSV → эмбеддинг через HuggingFace API → сохраняется в ChromaDB
2. **Запрос**: вопрос пользователя → эмбеддинг → поиск по косинусному сходству в ChromaDB → Top-5 релевантных фрагментов
3. **Генерация**: найденные фрагменты подставляются в промпт → LLM генерирует итоговый ответ

## Стек

| Компонент | Технология |
|-----------|-----------|
| API-сервер | FastAPI + Uvicorn |
| Векторная БД | ChromaDB (persistent, cosine similarity) |
| Эмбеддинги | HuggingFace Inference API — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| LLM | HuggingFace Inference API — `meta-llama/Llama-3.1-8B-Instruct` |
| Язык | Python 3.12+ |

Все модели — **бесплатный** доступ через HuggingFace Serverless Inference.

## Структура проекта

```
rag_chatbot/
├── config.py          # Конфигурация: модели, пути, параметры RAG
├── indexer.py         # Индексация CSV → ChromaDB (запускается один раз)
├── server.py          # FastAPI сервер с эндпоинтами /ask
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

### 1. Установка

```bash
git clone https://github.com/N-i-Kit-OS/rag-model && cd rag-model

pip install -r requirements.txt

export HF_TOKEN="hf_ваш_токен"
```

### 2. Индексация

```bash
python indexer.py
```

### 3. Запуск сервера

```bash
python server.py
```

### Задать вопрос

```
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Вопрсо?"}'
```

## Docker

### Сборка

```bash
docker build -t rag-model .
```

### Запуск

```bash
docker run -p 8000:8000 -e HF_TOKEN=hf_ваш_токен rag-model
```

## Лицензия

MIT
