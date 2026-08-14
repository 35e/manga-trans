# CPU-only image for the manga-trans API.
FROM python:3.12-slim-bookworm

# Retries *and* pipelining off — retries alone hit the same stall. See DOCS.md.
RUN printf 'Acquire::Retries "8";\nAcquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\n' \
        > /etc/apt/apt.conf.d/99robust

# Runtime libs for opencv-python-headless, plus a font to letter with.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MANGA_TRANS_MODEL=/opt/models/comictextdetector.pt.onnx \
    MANGA_TRANS_OCR_MODELS=/opt/models/ppocr \
    HF_HOME=/opt/models/hf

# First, so pip treats the requirement as met: the PyPI wheel drags the whole
# CUDA stack in. See DOCS.md.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2"

COPY requirements.txt /app/requirements.txt
# Three lines, not one: rapidocr pulls in the GUI cv2 and wins. See DOCS.md.
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

# Group 0 so the image also runs under `--user $(id -u):$(id -g)`. The models are
# fetched below as this user, into a directory it already owns: chowning them
# afterwards would copy every one into a second layer.
RUN useradd --uid 10001 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/models \
    && chown -R 10001:0 /opt/models \
    && chmod -R a+rX,g+rwX /opt/models

USER 10001

# --build-arg PREFETCH_MODEL=false fetches the weights on first use instead.
ARG PREFETCH_MODEL=true

# Only the six modules the fetchers need. Editing any of them invalidates
# ~810 MB of baked weights — do not copy the whole package here.
COPY mangatrans/__init__.py mangatrans/geometry.py mangatrans/languages.py \
     mangatrans/detect.py mangatrans/read.py mangatrans/inpaint.py /app/mangatrans/
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        python -c "from mangatrans.detect import ensure_model, ensure_regions; ensure_model(); ensure_regions()" \
        && python -c "from mangatrans.inpaint import ensure_model; ensure_model()" \
        && python -c "from mangatrans.read import ensure_readers; ensure_readers()"; \
    fi

COPY mangatrans /app/mangatrans

# No Ollama host here on purpose: `ollama.answering()` tries each name, since
# Docker and Podman disagree. MANGA_TRANS_OLLAMA still overrides it.
ENV MANGA_TRANS_HOST=0.0.0.0 \
    MANGA_TRANS_PORT=8000

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["python", "-m", "mangatrans"]
CMD []
