#!/usr/bin/env bash
# Run manga-trans in a container, building the image the first time.
#
#   ./run.sh                        # OCR every image in pages/ -> pages/out/
#   ./run.sh --translate            # ... and translate each bubble with ollama
#   ./run.sh 001.jpg --save-viz     # one page, plus an annotated copy
#   ./run.sh --help                 # every flag the script takes
#   ./run.sh test                   # unit tests (no models needed)
#   ./run.sh build                  # rebuild the image
#
# podman and docker both work; whichever is on PATH is used. Override with
# $CONTAINER_ENGINE, the ollama server with $OLLAMA_URL, the model with
# $OLLAMA_MODEL.
set -euo pipefail

cd "$(dirname "$0")"

engine=${CONTAINER_ENGINE:-}
if [ -z "$engine" ]; then
    for candidate in podman docker; do
        if command -v "$candidate" >/dev/null 2>&1; then
            engine=$candidate
            break
        fi
    done
fi
if [ -z "$engine" ]; then
    echo "run.sh: need podman or docker on PATH" >&2
    exit 1
fi

if [ "$engine" = "podman" ]; then
    image=${MANGA_TRANS_IMAGE:-localhost/manga-trans:latest}
    # Map the host user onto the container's uid/gid so files written to
    # pages/out/ belong to you rather than to root.
    id_args=(--userns="keep-id:uid=10001,gid=10001")
    host_alias=host.containers.internal
else
    image=${MANGA_TRANS_IMAGE:-manga-trans:latest}
    id_args=(--user "$(id -u):$(id -g)")
    host_alias=host.docker.internal
fi

build() {
    echo "run.sh: building $image (first run takes a few minutes)" >&2
    "$engine" build -t "$image" .
}

case "${1:-}" in
build)
    shift
    build
    [ $# -eq 0 ] && exit 0
    ;;
test)
    shift
    "$engine" image inspect "$image" >/dev/null 2>&1 || build
    # Mount the working copy so tests run against it without a rebuild.
    exec "$engine" run --rm --entrypoint python \
        -v "$PWD/manga_ocr_groups.py:/app/manga_ocr_groups.py:ro" \
        -v "$PWD/test_grouping.py:/app/test_grouping.py:ro" \
        "$image" /app/test_grouping.py "$@"
    ;;
esac

"$engine" image inspect "$image" >/dev/null 2>&1 || build

mkdir -p pages/out
mount=./pages:/pages
# SELinux hosts (Fedora/RHEL) refuse the mount without a relabel.
if command -v selinuxenabled >/dev/null 2>&1 && selinuxenabled 2>/dev/null; then
    mount=$mount:Z
fi

exec "$engine" run --rm --init \
    -v "$mount" \
    "${id_args[@]}" \
    -e "OLLAMA_URL=${OLLAMA_URL:-http://$host_alias:11434}" \
    ${OLLAMA_MODEL:+-e "OLLAMA_MODEL=$OLLAMA_MODEL"} \
    "$image" --cpu "$@"
