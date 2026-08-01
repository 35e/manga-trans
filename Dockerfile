# CPU-only image for manga_ocr_groups.py.
# Builds with either podman or docker - no BuildKit-only syntax is used.
FROM python:3.12-slim-bookworm

# Runtime libs for opencv-python-headless (pulled in by easyocr).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model caches live in the image, not in $HOME, so the container works
    # read-only-ish and with an arbitrary UID.
    EASYOCR_MODULE_PATH=/opt/models/easyocr \
    HF_HOME=/opt/models/huggingface

# CPU-only torch first, otherwise the easyocr dependency drags in ~4 GB of
# nvidia-* wheels that are useless in this image.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the models into the image (~530 MB) so a run needs no network.
# Build with --build-arg PREFETCH_MODELS=false to skip and download on first run.
ARG PREFETCH_MODELS=true
COPY scripts/prefetch_models.py /app/scripts/prefetch_models.py
RUN if [ "$PREFETCH_MODELS" = "true" ]; then \
        python /app/scripts/prefetch_models.py; \
    fi

COPY manga_ocr_groups.py test_grouping.py /app/

# uid 10001, group 0 so the image also runs under `--user $(id -u):$(id -g)`.
RUN useradd --uid 10001 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/models /pages \
    && chown -R 10001:0 /opt/models /pages \
    && chmod -R a+rX,g+rwX /opt/models /pages

USER 10001

# Manga pages are mounted here; relative paths on the command line resolve
# inside this directory. With no arguments the whole folder is read and one
# JSON per page is written to /pages/out.
# OLLAMA_URL points at the host's ollama (podman and docker both resolve these
# names; docker uses host.docker.internal).
ENV MANGA_TRANS_INPUT=/pages \
    MANGA_TRANS_OUT_DIR=/pages/out \
    OLLAMA_URL=http://host.containers.internal:11434 \
    OLLAMA_MODEL=gemma4:12b

# With the weights baked in, transformers must not phone home: it HEAD-requests
# the model files on every start, which costs a round-trip per run and fails
# outright with no network. Follows PREFETCH_MODELS ("true"/"false" are exactly
# what huggingface_hub parses), so a build without the models still downloads
# them on first run.
ENV HF_HUB_OFFLINE=${PREFETCH_MODELS}
WORKDIR /pages
VOLUME ["/pages"]

ENTRYPOINT ["python", "/app/manga_ocr_groups.py"]
# Empty, so a bare `run` processes the whole /pages folder. Must be set
# explicitly: otherwise the base image's CMD ["python3"] would be passed to the
# entrypoint as an argument.
CMD []
