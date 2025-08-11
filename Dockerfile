# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (опционально) curl для HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# сначала зависимости — лучше кэш
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# не-root юзер
RUN useradd -m -u 10001 appuser

# код и лексикон
COPY main.py ingredients.yml ./
RUN chown -R appuser:appuser /app
USER appuser

# дефолтные настройки (можно переопределить через -e)
ENV LEXICON_PATH=/app/ingredients.yml \
    FUZZY_THRESHOLD=90 \
    MAX_ITEMS=200 \
    PORT=8000 \
    WEB_CONCURRENCY=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# uvicorn с переменными окружения (кол-во воркеров настраивается WEB_CONCURRENCY)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
