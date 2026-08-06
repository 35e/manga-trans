# CPU-only image for the manga-trans API.
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
    MANGA_TRANS_MODEL=/opt/models/comictextdetector.pt.onnx

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the detector (~95 MB) into the image so a run needs no network.
# Build with --build-arg PREFETCH_MODEL=false to fetch it on first use instead.
ARG PREFETCH_MODEL=true
COPY mangatrans /app/mangatrans
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        python -c "from mangatrans.detect import ensure_model; ensure_model()"; \
    fi

# uid 10001, group 0 so the image also runs under `--user $(id -u):$(id -g)`.
RUN useradd --uid 10001 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/models \
    && chown -R 10001:0 /opt/models \
    && chmod -R a+rX,g+rwX /opt/models

USER 10001

ENV MANGA_TRANS_HOST=0.0.0.0 \
    MANGA_TRANS_PORT=8000

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["python", "-m", "mangatrans"]
CMD []
