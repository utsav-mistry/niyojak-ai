# syntax=docker/dockerfile:1

# Stage 1: train the model
# Runs train_model.py to produce niyojak_model.pkl inside the image.
# If you want to supply a pre-trained model instead, you can skip this stage
# and COPY the .pkl file directly in stage 2.
FROM python:3.11-slim AS trainer

WORKDIR /train

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/  ./app/
COPY train/ ./train/

# Run training — produces model/niyojak_model.pkl
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

# Set environment defaults (overridable at runtime via K8s env vars or Helm values)
ENV PORT=8000 \
    PROMETHEUS_URL=http://prometheus:9090 \
    MODEL_PATH=/app/model/niyojak_model.pkl \
    POLL_INTERVAL_SEC=5 \
    WINDOW_SIZE=12

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

WORKDIR /app/app
CMD ["python", "main.py"]
