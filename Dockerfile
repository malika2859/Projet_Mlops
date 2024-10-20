# Image de base Python slim
FROM python:3.10-slim

# Installation des dépendances système avec nettoyage dans la même couche
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configuration de l'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GUNICORN_CMD_ARGS="--log-level=debug"

WORKDIR /app

# Copie et installation des requirements en plusieurs étapes
COPY requirements.txt .

# Installation des dépendances scientifiques en premier
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    pandas==2.0.0 \
    scipy==1.10.1

# Installation des autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du modèle séparément pour profiter du cache des couches
COPY random_forest_model.pkl .

# Vérification du modèle
RUN ls -la /app/random_forest_model.pkl || echo "ATTENTION: Modèle non trouvé!"

# Copie du code de l'application
COPY . .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "120", "--log-level", "debug", "--capture-output", "--enable-stdio-inheritance", "app:app"]