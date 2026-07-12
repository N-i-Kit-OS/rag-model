FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py server.py ./

COPY chroma_db/ ./chroma_db/

EXPOSE 8000

CMD ["python", "server.py"]
