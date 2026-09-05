"""Serve the API: python -m mangatrans"""

from __future__ import annotations

import os

from waitress import serve

from .server import MAX_UPLOAD, create_app


def main() -> None:
    host = os.environ.get("MANGA_TRANS_HOST", "127.0.0.1")
    port = int(os.environ.get("MANGA_TRANS_PORT", "8000"))
    print(f"manga-trans api on http://{host}:{port}/api", flush=True)
    serve(create_app(), host=host, port=port, max_request_body_size=MAX_UPLOAD)


if __name__ == "__main__":
    main()
