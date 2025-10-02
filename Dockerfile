FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY predict_server.py ./predict_server.py
COPY models/deeppavlov ./models/deeppavlov

ENV MODEL_DIR=/app/models/deeppavlov

EXPOSE 8000
#predict_server:app не забудь поменять алёё
CMD ["uvicorn", "predict_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
