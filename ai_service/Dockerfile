# syntax=docker/dockerfile:1

# Stage 1: runtime image only.
# The model (niyojak_model.json) is pre-trained offline and COPY-ed in below.
# To rebuild with a fresh model: run train/train_model.py locally first,
# then rebuild this image so the new model/niyojak_model.json is picked up.
FROM python:3.11-slim AS trainer

WORKDIR /train

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/  ./app/
COPY train/ ./train/

# Run training — produces model/niyojak_model.json
RUN python train/train_model.py


# Stage 2: runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy trained model from the trainer stage
COPY --from=trainer /train/model/ ./model/

# Set environment defaults (overridable at runtime via K8s env vars)
ENV PORT=8000 \
    K8S_API_URL=https://kubernetes.default.svc \
    MODEL_PATH=/app/model/niyojak_model.json \
    POLL_INTERVAL_SEC=5

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

WORKDIR /app/app
CMD ["python", "main.py"]
