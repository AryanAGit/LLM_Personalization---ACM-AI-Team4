import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from enron_style import (
    EmailRecord,
    describe_style_heuristic,
    generate_style_response,
    load_history,
    normalize_backend,
    score_against_profile,
    score_prediction,
)


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_HISTORY = ROOT / "data" / "authors" / "lincoln" / "user_email_history.json"
DEFAULT_PROFILES = ROOT / "data" / "processed" / "profile_user.json"


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StyleLabHandler(BaseHTTPRequestHandler):
    histories = []
    profiles_by_id = {}
    model = "llama3.1:8b"
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = ""
    adapter_root = "data/lora_adapters"

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
            user_id = str(payload.get("user_id") or self.histories[0]["user_id"])
            use_ollama = bool(payload.get("use_ollama", True))
            backend = str(payload.get("backend") or ("ollama" if use_ollama else "fallback"))
            backend = normalize_backend(backend, use_ollama)
            model = str(payload.get("model") or self.model)
            base_model = str(payload.get("base_model") or self.base_model)
            adapter_path = str(payload.get("adapter_path") or self.adapter_path)
            adapter_root = str(payload.get("adapter_root") or self.adapter_root)
        except (ValueError, IndexError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Invalid request: {exc}"}, status=400)
            return

        if not prompt:
            self.send_json({"error": "Prompt is required."}, status=400)
            return

        selected = self.find_user(user_id)
        if not selected:
            self.send_json({"error": f"Unknown user_id: {user_id}"}, status=404)
            return

        try:
            output = generate_style_response(
                profile=selected["profile"],
                prompt=prompt,
                use_ollama=use_ollama,
                model=model,
                backend=backend,
                base_model=base_model,
                adapter_path=adapter_path,
                adapter_root=adapter_root,
                user_id=user_id,
                identity=selected.get("inferred_name", ""),
            )
            effective_backend = backend
            warning = ""
        except Exception as exc:
            if backend != "peft" or not is_lora_quality_failure(exc):
                self.send_json({"error": str(exc)}, status=500)
                return
            output = generate_style_response(
                profile=selected["profile"],
                prompt=prompt,
                use_ollama=False,
                model=model,
                backend="fallback",
                identity=selected.get("inferred_name", ""),
            )
            effective_backend = "fallback"
            warning = "LoRA output failed quality checks; showing fallback/RAG output."

        self.send_json(
            {
                "output": output,
                "user": self.serialize_user(selected),
                "model": model,
                "base_model": base_model,
                "adapter_path": adapter_path,
                "adapter_root": adapter_root,
                "backend": effective_backend,
                "requested_backend": backend,
                "warning": warning,
                "used_ollama": effective_backend == "ollama",
            }
        )

    def handle_test(self) -> None:
        try:
            payload = self.read_json()
            user_id = str(payload.get("user_id") or self.histories[0]["user_id"])
            query_id = str(payload.get("query_id", "")).strip()
            use_ollama = bool(payload.get("use_ollama", True))
            backend = str(payload.get("backend") or ("ollama" if use_ollama else "fallback"))
            backend = normalize_backend(backend, use_ollama)
            model = str(payload.get("model") or self.model)
            base_model = str(payload.get("base_model") or self.base_model)
            adapter_path = str(payload.get("adapter_path") or self.adapter_path)
            adapter_root = str(payload.get("adapter_root") or self.adapter_root)
        except (ValueError, IndexError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Invalid request: {exc}"}, status=400)
            return

        selected = self.find_user(user_id)
        if not selected:
            self.send_json({"error": f"Unknown user_id: {user_id}"}, status=404)
            return

        query = self.find_query(selected, query_id)
        if not query:
            self.send_json({"error": f"Unknown query_id: {query_id}"}, status=404)
            return

        try:
            output = generate_style_response(
                profile=selected["profile"],
                prompt=query["input"],
                use_ollama=use_ollama,
                model=model,
                backend=backend,
                base_model=base_model,
                adapter_path=adapter_path,
                adapter_root=adapter_root,
                user_id=user_id,
                identity=selected.get("inferred_name", ""),
            )
            scores = score_prediction(output, query["gold"])
            scores.update(score_against_profile(output, selected.get("profile", [])))
            effective_backend = backend
            warning = ""
        except Exception as exc:
            if backend != "peft" or not is_lora_quality_failure(exc):
                self.send_json({"error": str(exc)}, status=500)
                return
            output = generate_style_response(
                profile=selected["profile"],
                prompt=query["input"],
                use_ollama=False,
                model=model,
                backend="fallback",
                identity=selected.get("inferred_name", ""),
            )
            scores = score_prediction(output, query["gold"])
            scores.update(score_against_profile(output, selected.get("profile", [])))
            effective_backend = "fallback"
            warning = "LoRA output failed quality checks; showing fallback/RAG output."

        self.send_json(
            {
                "query": self.serialize_query(query),
                "generated": output,
                "actual": query["gold"],
                "scores": scores,
                "user": self.serialize_user(selected),
                "model": model,
                "base_model": base_model,
                "adapter_path": adapter_path,
                "adapter_root": adapter_root,
                "backend": effective_backend,
                "requested_backend": backend,
                "warning": warning,
                "used_ollama": effective_backend == "ollama",
            }
        )

    def find_user(self, user_id) -> Optional[dict]:
        return next((item for item in self.histories if str(item["user_id"]) == str(user_id)), None)

    def find_query(self, history: dict, query_id: str) -> Optional[dict]:
        queries = history.get("query", [])
        if not query_id and queries:
            return queries[0]
        return next((item for item in queries if item["id"] == query_id), None)

    def serialize_query(self, query: dict) -> dict:
        return {
            "id": query["id"],
            "input": query["input"],
            "gold": query["gold"],
            "subject": extract_subject_from_input(query["input"]),
            "has_context": "Incoming email:" in query["input"],
            "gold_word_count": len(query["gold"].split()),
        }

    def serialize_users(self) -> list:
        return [self.serialize_user(history) for history in self.histories]

    def serialize_user(self, history: dict) -> dict:
        user_id = history["user_id"]
        return {
            "user_id": user_id,
            "source_user": history.get("source_user", ""),
            "inferred_name": history.get("inferred_name", ""),
            "profile_count": len(history.get("profile", [])),
            "query_count": len(history.get("query", [])),
            "style": self.profiles_by_id.get(user_id, "") or describe_profile_style(history),
            "queries": [self.serialize_query(query) for query in history.get("query", [])],
        }

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


def load_profiles(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    return {item["id"]: item.get("output", "") for item in profiles}


def describe_profile_style(history: dict) -> str:
    records = [
        EmailRecord(
            id=item.get("id", ""),
            user_name=str(history.get("user_id", "")),
            subject=item.get("subject", ""),
            body=item.get("body", ""),
        )
        for item in history.get("profile", [])
    ]
    summary = describe_style_heuristic(records)
    return (
        summary.replace("The user's emails are", "The source passages are")
        .replace("They usually gets", "They usually get")
        .replace("and uses", "and use")
        .replace("Their messages average", "The passages average")
    )


def is_lora_quality_failure(exc: Exception) -> bool:
    message = str(exc)
    return "LoRA model returned unusable text" in message or "PEFT/LoRA generation failed" in message


def extract_subject_from_input(text: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip() or "(no subject)"
    for marker in ["inspired by this topic:", "for this subject:"]:
        if marker in text:
            subject = text.split(marker, 1)[1].strip()
            subject = subject.split("Do not quote", 1)[0].strip().strip(".")
            return subject or "(no topic)"
    return "(no subject)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the style generation web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--adapter-root", default="data/lora_adapters")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the app in the default browser after the server starts.",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history JSON: {history_path}")

    StyleLabHandler.histories = load_history(history_path)
    StyleLabHandler.profiles_by_id = load_profiles(Path(args.profiles))
    StyleLabHandler.model = args.model
    StyleLabHandler.base_model = args.base_model
    StyleLabHandler.adapter_path = args.adapter_path
    StyleLabHandler.adapter_root = args.adapter_root

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
