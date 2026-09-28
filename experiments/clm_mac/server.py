"""Local HTTP adapter exposing the cached Mac CLM engine to AutoInsight."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from demo import CACHE_DIR, CANDIDATES, ENCODER, HEAD_REPO, MacEmbedder, preflight


class ClmService:
    def __init__(self, cache_dir: Path, local_only: bool, max_tokens: int) -> None:
        from clm import Engine
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            HEAD_REPO,
            "CLM_v0.1-8B.pt",
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        embedder = MacEmbedder(cache_dir, max_tokens, local_only)
        self.engine = Engine(embedder=embedder, checkpoint=checkpoint, device="cpu", action_cache=0)
        self.lock = threading.Lock()

    def rank(self, body: dict) -> dict:
        question = str(body.get("question", ""))
        candidates = body.get("candidates", CANDIDATES)
        if not question or not isinstance(candidates, list) or not candidates:
            raise ValueError("question and a non-empty candidates list are required")
        state = body.get("state", "AutoInsight vehicle analysis")
        with self.lock:
            ranked = self.engine.rank(
                str(state),
                [str(candidate) for candidate in candidates],
                instructions="Which allowed analysis action best addresses the user's question?",
            )
        return {"status": "live", "backend": "clm_mac_mps", "encoder": ENCODER, "ranked": ranked}


class Handler(BaseHTTPRequestHandler):
    service: ClmService | None = None

    def _write(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write(200, {"status": "ok", "model_loaded": self.service is not None})
        else:
            self._write(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/rank" or self.service is None:
            self._write(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            self._write(200, self.service.rank(body))
        except (ValueError, json.JSONDecodeError) as exc:
            self._write(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - protects the long-running service
            self._write(500, {"error": f"CLM inference failed: {exc}"})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    check = preflight(args.cache_dir)
    if not check["mps_available"]:
        raise SystemExit(
            "Apple MPS is unavailable; run the CLM server on the configured Mac runtime."
        )
    print("Loading CLM weights; cached startup can still take several minutes...", flush=True)
    Handler.service = ClmService(args.cache_dir, args.local_only, args.max_tokens)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CLM server ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
