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
    MANGA_TRANS_MODEL=/opt/models/comictextdetector.pt.onnx \
    HF_HOME=/opt/models/hf

# manga-ocr wants torch. On x86_64 the wheel on PyPI drags the whole CUDA stack
# in — gigabytes of it, for a card this image will never see. PyTorch's CPU
# index has the same torch without any of that, so it goes in first and pip
# treats the requirement as already met.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# uid 10001, group 0 so the image also runs under `--user $(id -u):$(id -g)`.
# The models are fetched below as this user, into a directory it already owns:
# chowning half a gigabyte of weights afterwards would copy every one of them
# into a second layer.
RUN useradd --uid 10001 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/models \
    && chown -R 10001:0 /opt/models \
    && chmod -R a+rX,g+rwX /opt/models

USER 10001

# Bake both models into the image so a run needs no network: the detector
# (~95 MB) and the reader (~450 MB).
# Build with --build-arg PREFETCH_MODEL=false to fetch them on first use instead.
ARG PREFETCH_MODEL=true
COPY mangatrans /app/mangatrans
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        python -c "from mangatrans.detect import ensure_model; ensure_model()" \
        && python -c "from mangatrans.read import ensure_model; ensure_model()"; \
    fi

ENV MANGA_TRANS_HOST=0.0.0.0 \
    MANGA_TRANS_PORT=8000

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["python", "-m", "mangatrans"]
CMD []
