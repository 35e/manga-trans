# CPU-only image for the manga-trans API.
FROM python:3.12-slim-bookworm

# Debian's package index is ~12 MB in one response, and a container network that
# carries small files fine — a VM on a Mac, most proxies — drops that one part
# way through, which apt reports as a mirror failure rather than a stall. Retries
# alone do not fix it; the pipelining has to go too, or the retry hits the same
# stall. Cheap insurance on a build that otherwise cannot start.
RUN printf 'Acquire::Retries "8";\nAcquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\n' \
        > /etc/apt/apt.conf.d/99robust

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
    MANGA_TRANS_OCR_MODELS=/opt/models/ppocr \
    HF_HOME=/opt/models/hf

# manga-ocr wants torch. On x86_64 the wheel on PyPI drags the whole CUDA stack
# in — gigabytes of it, for a card this image will never see. PyTorch's CPU
# index has the same torch without any of that, so it goes in first and pip
# treats the requirement as already met.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2"

COPY requirements.txt /app/requirements.txt
# RapidOCR asks for opencv-python, which is the same cv2 as the headless build
# but linked against a libGL this image has no use for — and installed second, it
# is the one that wins. Dropping it and laying the headless build back down
# leaves one cv2 that imports without a display, which is what everything here
# was using already.
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

# uid 10001, group 0 so the image also runs under `--user $(id -u):$(id -g)`.
# The models are fetched below as this user, into a directory it already owns:
# chowning half a gigabyte of weights afterwards would copy every one of them
# into a second layer.
RUN useradd --uid 10001 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/models \
    && chown -R 10001:0 /opt/models \
    && chmod -R a+rX,g+rwX /opt/models

USER 10001

# Bake every model into the image so a run needs no network: the region detector
# (~44 MB), the ink mask (~95 MB), LaMa (~206 MB), manga-ocr (~450 MB) and
# PP-OCR's recognisers (a few MB per language).
# Build with --build-arg PREFETCH_MODEL=false to fetch them on first use instead.
ARG PREFETCH_MODEL=true

# Only the modules the fetchers themselves need. Copying the whole package here
# instead would send every later build back for the better part of a gigabyte of
# weights over an edit to the server. Adding a language to `languages.py` costs
# that too, which is honest: a new language is new weights.
COPY mangatrans/__init__.py mangatrans/geometry.py mangatrans/languages.py \
     mangatrans/detect.py mangatrans/read.py mangatrans/inpaint.py /app/mangatrans/
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        python -c "from mangatrans.detect import ensure_model, ensure_regions; ensure_model(); ensure_regions()" \
        && python -c "from mangatrans.inpaint import ensure_model; ensure_model()" \
        && python -c "from mangatrans.read import ensure_readers; ensure_readers()"; \
    fi

COPY mangatrans /app/mangatrans

# Translation asks a model on the machine running the container, not one inside
# it. What that machine is called from in here is Docker's or Podman's to say and
# they do not agree, so mangatrans.ollama tries each name rather than one being
# baked in; MANGA_TRANS_OLLAMA still overrides it.
ENV MANGA_TRANS_HOST=0.0.0.0 \
    MANGA_TRANS_PORT=8000

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["python", "-m", "mangatrans"]
CMD []
