FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Sao_Paulo

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN useradd -m appuser
USER appuser

EXPOSE 8000

ENTRYPOINT ["./start.sh"]
