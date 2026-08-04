#!/usr/bin/env bash
# Run the manga-trans GUI in a container, building the image the first time.
#
#   ./run.sh            # open http://localhost:8000
#   ./run.sh test       # unit tests (no models needed)
#   ./run.sh build      # rebuild the image after changing the code
#
# podman and docker both work; whichever is on PATH is used. Override with
# $CONTAINER_ENGINE, the port with $PORT, the ollama server with $OLLAMA_URL
# and the model with $OLLAMA_MODEL.
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
    # out/ belong to you rather than to root.
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
    build
    exit 0
    ;;
test)
    shift
    "$engine" image inspect "$image" >/dev/null 2>&1 || build
    exec "$engine" run --rm --entrypoint python -w /app \
        -v "$PWD/mangatrans:/app/mangatrans:ro" \
        -v "$PWD/tests:/app/tests:ro" \
        "$image" -m unittest discover -s tests -t . "$@"
    ;;
esac

"$engine" image inspect "$image" >/dev/null 2>&1 || build

port=${PORT:-8000}
mkdir -p pages out
suffix=
# SELinux hosts (Fedora/RHEL) refuse the mounts without a relabel.
if command -v selinuxenabled >/dev/null 2>&1 && selinuxenabled 2>/dev/null; then
    suffix=:Z
fi

echo "run.sh: open http://localhost:$port" >&2
exec "$engine" run --rm --init \
    -p "$port:8000" \
    -v "./pages:/pages$suffix" \
    -v "./out:/out$suffix" \
    "${id_args[@]}" \
    -e "OLLAMA_URL=${OLLAMA_URL:-http://$host_alias:11434}" \
    ${OLLAMA_MODEL:+-e "OLLAMA_MODEL=$OLLAMA_MODEL"} \
    "$image" "$@"
