# =========================
# STAGE 1 - Builder
# =========================
FROM python:3.9-alpine AS builder

WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .

# Installer les dépendances dans le builder
RUN pip install --no-cache-dir -r requirements.txt


# =========================
# STAGE 2 - Final Image
# =========================
FROM python:3.9-alpine

WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et le modèle
COPY app.py .
COPY model/ ./model/

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
