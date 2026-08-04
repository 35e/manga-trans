# CPU-only image for the manga-trans review GUI.
FROM python:3.12-slim-bookworm

# Runtime libs for opencv-python-headless, plus a font: the slim image ships
# none at all, and the overlay needs one to letter with.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/opt/models/huggingface

# CPU-only torch first, otherwise manga-ocr drags in ~4 GB of nvidia-* wheels
# that are useless here. Only recognition needs torch; detection runs on
# OpenCV's ONNX backend.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the models into the image (~545 MB) so a run needs no network.
# Build with --build-arg PREFETCH_MODELS=false to download them on first use.
ARG PREFETCH_MODELS=true
COPY scripts /app/scripts
COPY mangatrans /app/mangatrans
RUN if [ "$PREFETCH_MODELS" = "true" ]; then python /app/scripts/fetch_models.py; fi

# uid 10001, group 0 so the image also runs under `--user $(id -u):$(id -g)`.
RUN useradd --uid 10001 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/models /pages /out \
    && chown -R 10001:0 /opt/models /pages /out \
    && chmod -R a+rX,g+rwX /opt/models /pages /out

USER 10001

# With the weights baked in, transformers must not phone home: it HEAD-requests
# the model files on every start, which fails outright with no network.
ENV MANGA_TRANS_PAGES=/pages \
    MANGA_TRANS_OUT=/out \
    MANGA_TRANS_HOST=0.0.0.0 \
    MANGA_TRANS_PORT=8000 \
    OLLAMA_URL=http://host.containers.internal:11434 \
    OLLAMA_MODEL=gemma4:12b \
    HF_HUB_OFFLINE=${PREFETCH_MODELS}

WORKDIR /app
VOLUME ["/pages", "/out"]
EXPOSE 8000

ENTRYPOINT ["python", "-m", "mangatrans"]
CMD []
