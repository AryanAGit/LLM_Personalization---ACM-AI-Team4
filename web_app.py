import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from enron_style import generate_style_response

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_HISTORY = ROOT / "trump_data" / "trump_tweets.jsonl"


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StyleLabHandler(BaseHTTPRequestHandler):
    histories = []
    queries = []
    model = "llama3.1:8b"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/users":
            self.send_json({"users": self.serialize_users()})
            return
        if parsed.path == "/":
            self.serve_file(WEB_ROOT / "index.html")
            return
        self.serve_file(WEB_ROOT / parsed.path.lstrip("/"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            self.handle_generate()
            return
        if parsed.path == "/api/test":
            self.handle_test()
            return
        self.send_error(404)

    def handle_generate(self) -> None:
        try:
            payload = self.read_json()
            prompt = str(payload.get("prompt", "")).strip()
            use_ollama = bool(payload.get("use_ollama", True))
            model = str(payload.get("model") or self.model)
        except (ValueError, IndexError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Invalid request: {exc}"}, status=400)
            return

        if not prompt:
            self.send_json({"error": "Prompt is required."}, status=400)
            return

        try:
            output = generate_style_response(
                profile=[],
                prompt=prompt,
                use_ollama=use_ollama,
                model=model,
                identity="Donald Trump",
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)
            return

        self.send_json(
            {
                "output": output,
                "model": model,
                "used_ollama": use_ollama,
            }
        )

    def handle_test(self) -> None:
        try:
            payload = self.read_json()
            query_id = str(payload.get("query_id", "")).strip()
            use_ollama = bool(payload.get("use_ollama", True))
            model = str(payload.get("model") or self.model)
        except (ValueError, IndexError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Invalid request: {exc}"}, status=400)
            return

        query = self.find_query(query_id)
        if not query:
            self.send_json({"error": f"Unknown query_id: {query_id}"}, status=404)
            return

        try:
            output = generate_style_response(
                profile=[],
                prompt=query["input"],
                use_ollama=use_ollama,
                model=model,
                identity="Donald Trump",
            )
            # Remove score_prediction entirely since personalization/scoring is removed
            scores = {}
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)
            return

        self.send_json(
            {
                "query": self.serialize_query(query),
                "generated": output,
                "actual": query["gold"],
                "scores": scores,
                "model": model,
                "used_ollama": use_ollama,
            }
        )

    def find_query(self, query_id: str) -> Optional[dict]:
        if not query_id and self.queries:
            return self.queries[0]
        return next((item for item in self.queries if item["id"] == query_id), None)

    def serialize_query(self, query: dict) -> dict:
        return {
            "id": query["id"],
            "input": query["input"],
            "gold": query["gold"],
            "subject": query["input"][:50] + "...",
            "has_context": False,
            "gold_word_count": len(query["gold"].split()),
        }

    def serialize_users(self) -> list:
        # We only have one user: Donald Trump
        return [{
            "user_id": 1,
            "source_user": "realDonaldTrump",
            "inferred_name": "Donald Trump",
            "profile_count": 0,
            "query_count": len(self.queries),
            "base_model": self.model,
            "style": "",
            "queries": [self.serialize_query(query) for query in self.queries],
        }]

    def read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            if not str(resolved).startswith(str(WEB_ROOT.resolve())) or not resolved.is_file():
                self.send_error(404)
                return
            content = resolved.read_bytes()
        except OSError:
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, status: int = 200) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def load_trump_history(path: Path) -> list:
    if not path.exists():
        return []
    
    queries = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip(): continue
            data = json.loads(line)
            queries.append({
                "id": str(idx),
                "input": data.get("prompt", ""),
                "gold": data.get("response", "")
            })
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Enron style generation web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the app in the default browser after the server starts.",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history JSON: {history_path}")

    StyleLabHandler.queries = load_trump_history(history_path)
    StyleLabHandler.model = args.model

    server, port = create_server(args.host, args.port)
    url = f"http://{args.host}:{port}"
    print(f"Style Lab running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


def create_server(host: str, preferred_port: int, attempts: int = 20) -> Tuple[ThreadingHTTPServer, int]:
    for port in range(preferred_port, preferred_port + attempts):
        try:
            return ReusableThreadingHTTPServer((host, port), StyleLabHandler), port
        except OSError as exc:
            if exc.errno != 48:
                raise
            print(f"Port {port} is already in use; trying {port + 1}.")
    raise OSError(f"No open port found from {preferred_port} to {preferred_port + attempts - 1}.")


if __name__ == "__main__":
    main()
