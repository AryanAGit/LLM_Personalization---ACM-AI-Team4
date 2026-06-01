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
GENERATED_HISTORY = ROOT / "data" / "processed" / "user_email_history.json"
GENERATED_PROFILES = ROOT / "data" / "processed" / "profile_user.json"
DEMO_HISTORY = ROOT / "lamp3_user_email_history.json"
DEMO_PROFILES = ROOT / "lamp3_profile_user.json"
PER_USER_HISTORY = ROOT / "per_user.json"
DEFAULT_HISTORY = PER_USER_HISTORY
DEFAULT_PROFILES = DEMO_PROFILES
DEFAULT_HF_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_HF_ADAPTER_REPO = "alchin2/lora-project"
HF_MODEL_PRESETS = [
    {
        "id": "hf_obama",
        "label": "Barack Obama",
        "source_user": "Hugging Face: Obama_v2",
        "base_model": DEFAULT_HF_BASE_MODEL,
        "adapter_path": DEFAULT_HF_ADAPTER_REPO,
        "adapter_subfolder": "Obama_v2",
        "style": "Team LoRA adapter trained for Barack Obama-style public speech.",
    },
    {
        "id": "hf_trump",
        "label": "Donald Trump",
        "source_user": "Hugging Face: trump",
        "base_model": DEFAULT_HF_BASE_MODEL,
        "adapter_path": DEFAULT_HF_ADAPTER_REPO,
        "adapter_subfolder": "trump",
        "style": "Team LoRA adapter trained for Donald Trump-style short-form public posts.",
    },
    {
        "id": "hf_twain",
        "label": "Mark Twain",
        "source_user": "Hugging Face: Twain_v1",
        "base_model": DEFAULT_HF_BASE_MODEL,
        "adapter_path": DEFAULT_HF_ADAPTER_REPO,
        "adapter_subfolder": "Twain_v1",
        "style": "Team LoRA adapter trained for Mark Twain-style literary prose.",
    },
    {
        "id": "hf_enron",
        "label": "Enron",
        "source_user": "Hugging Face: Enron",
        "base_model": DEFAULT_HF_BASE_MODEL,
        "adapter_path": DEFAULT_HF_ADAPTER_REPO,
        "adapter_subfolder": "Enron",
        "style": "Team LoRA adapter trained on Enron-style business email data.",
        "validation_source": "per_user",
    },
    {
        "id": "hf_jefferson",
        "label": "Thomas Jefferson",
        "source_user": "Hugging Face: Jefferson_Model",
        "base_model": DEFAULT_HF_BASE_MODEL,
        "adapter_path": DEFAULT_HF_ADAPTER_REPO,
        "adapter_subfolder": "Jefferson_Model",
        "style": "Team LoRA adapter trained for Thomas Jefferson-style historical prose.",
    },
]


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StyleLabHandler(BaseHTTPRequestHandler):
    histories = []
    profiles_by_id = {}
    model = "llama3.1:8b"
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = ""
    adapter_root = ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/users":
            self.send_json({"users": self.serialize_users()})
            return
        if parsed.path == "/api/model-presets":
            self.send_json({"presets": HF_MODEL_PRESETS})
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
        if parsed.path == "/api/compare":
            self.handle_compare()
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
            adapter_subfolder = str(payload.get("adapter_subfolder") or "")
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
        base_model, adapter_path, adapter_subfolder = self.resolve_hf_settings(
            selected,
            base_model,
            adapter_path,
            adapter_subfolder,
        )

        try:
            output = generate_style_response(
                profile=selected["profile"],
                prompt=prompt,
                use_ollama=use_ollama,
                model=model,
                backend=backend,
                base_model=base_model,
                adapter_path=adapter_path,
                adapter_subfolder=adapter_subfolder,
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
                "adapter_subfolder": adapter_subfolder,
                "adapter_root": adapter_root,
                "backend": effective_backend,
                "requested_backend": backend,
                "warning": warning,
                "used_ollama": effective_backend == "ollama",
            }
        )

    def handle_compare(self) -> None:
        try:
            payload = self.read_json()
            prompt = str(payload.get("prompt", "")).strip()
            user_id = str(payload.get("user_id") or self.histories[0]["user_id"])
            model = str(payload.get("model") or self.model)
            base_model = str(payload.get("base_model") or self.base_model)
            adapter_path = str(payload.get("adapter_path") or self.adapter_path)
            adapter_subfolder = str(payload.get("adapter_subfolder") or "")
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
        base_model, adapter_path, adapter_subfolder = self.resolve_hf_settings(
            selected,
            base_model,
            adapter_path,
            adapter_subfolder,
        )

        base_result = self.try_generate(
            selected=selected,
            prompt=prompt,
            model=model,
            backend="peft",
            base_model=base_model,
            adapter_path="",
            adapter_subfolder="",
            adapter_root="",
            user_id=user_id,
        )
        lora_result = self.try_generate(
            selected=selected,
            prompt=prompt,
            model=model,
            backend="peft",
            base_model=base_model,
            adapter_path=adapter_path,
            adapter_subfolder=adapter_subfolder,
            adapter_root=adapter_root,
            user_id=user_id,
        )
        self.send_json(
            {
                "user": self.serialize_user(selected),
                "prompt": prompt,
                "model": model,
                "base_model": base_model,
                "adapter_path": adapter_path,
                "adapter_subfolder": adapter_subfolder,
                "adapter_root": adapter_root,
                "base": base_result,
                "lora": lora_result,
            }
        )

    def try_generate(
        self,
        selected: dict,
        prompt: str,
        model: str,
        backend: str,
        base_model: str,
        adapter_path: str,
        adapter_subfolder: str,
        adapter_root: str,
        user_id: str,
    ) -> dict:
        try:
            output = generate_style_response(
                profile=selected["profile"],
                prompt=prompt,
                use_ollama=False,
                model=model,
                backend=backend,
                base_model=base_model,
                adapter_path=adapter_path,
                adapter_subfolder=adapter_subfolder,
                adapter_root=adapter_root,
                user_id=user_id,
                identity=selected.get("inferred_name", ""),
            )
            return {"ok": True, "output": output}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

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
            adapter_subfolder = str(payload.get("adapter_subfolder") or "")
            adapter_root = str(payload.get("adapter_root") or self.adapter_root)
        except (ValueError, IndexError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Invalid request: {exc}"}, status=400)
            return

        selected = self.find_user(user_id)
        if not selected:
            self.send_json({"error": f"Unknown user_id: {user_id}"}, status=404)
            return
        base_model, adapter_path, adapter_subfolder = self.resolve_hf_settings(
            selected,
            base_model,
            adapter_path,
            adapter_subfolder,
        )

        query = self.find_query(selected, query_id)
        if not query:
            if not selected.get("query"):
                self.send_json(
                    {"error": f"No held-out validation examples are available for {selected.get('inferred_name', 'this voice')}."},
                    status=400,
                )
            else:
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
                adapter_subfolder=adapter_subfolder,
                adapter_root=adapter_root,
                user_id=user_id,
                identity=selected.get("inferred_name", ""),
            )
            scores = score_prediction(output, query["gold"], profile=selected["profile"])
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
                "adapter_subfolder": adapter_subfolder,
                "adapter_root": adapter_root,
                "backend": effective_backend,
                "requested_backend": backend,
                "warning": warning,
                "used_ollama": effective_backend == "ollama",
            }
        )

    def find_user(self, user_id) -> Optional[dict]:
        preset = self.find_hf_preset(user_id)
        if preset:
            validation_history = self.validation_history_for_preset(preset)
            return {
                "user_id": preset["id"],
                "source_user": preset["source_user"],
                "inferred_name": preset["label"],
                "profile": validation_history.get("profile", []),
                "query": validation_history.get("query", []),
                "hf_preset": preset,
            }
        return next((item for item in self.histories if str(item["user_id"]) == str(user_id)), None)

    def find_hf_preset(self, preset_id: str) -> Optional[dict]:
        return next((preset for preset in HF_MODEL_PRESETS if preset["id"] == str(preset_id)), None)

    def resolve_hf_settings(
        self,
        selected: dict,
        base_model: str,
        adapter_path: str,
        adapter_subfolder: str,
    ) -> Tuple[str, str, str]:
        preset = selected.get("hf_preset")
        if not preset:
            return base_model, adapter_path, adapter_subfolder
        return (
            preset.get("base_model", base_model),
            preset.get("adapter_path", adapter_path),
            preset.get("adapter_subfolder", adapter_subfolder),
        )

    def validation_history_for_preset(self, preset: dict) -> dict:
        if preset.get("validation_source") != "per_user" or not self.histories:
            return {"profile": [], "query": []}
        return self.histories[0]

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
        return [self.serialize_preset_user(preset) for preset in HF_MODEL_PRESETS]

    def serialize_preset_user(self, preset: dict) -> dict:
        validation_history = self.validation_history_for_preset(preset)
        return {
            "user_id": preset["id"],
            "source_user": preset["source_user"],
            "inferred_name": preset["label"],
            "profile_count": len(validation_history.get("profile", [])),
            "query_count": len(validation_history.get("query", [])),
            "style": preset["style"],
            "queries": [self.serialize_query(query) for query in validation_history.get("query", [])],
            "hf_preset": preset,
        }

    def serialize_user(self, history: dict) -> dict:
        preset = history.get("hf_preset")
        user_id = history["user_id"]
        return {
            "user_id": user_id,
            "source_user": history.get("source_user", ""),
            "inferred_name": history.get("inferred_name", ""),
            "profile_count": len(history.get("profile", [])),
            "query_count": len(history.get("query", [])),
            "style": (preset or {}).get("style")
            or self.profiles_by_id.get(user_id, "")
            or describe_profile_style(history),
            "queries": [self.serialize_query(query) for query in history.get("query", [])],
            **({"hf_preset": preset} if preset else {}),
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


def load_web_history(path: Path) -> list:
    histories = load_history(path)
    if histories and "examples" in histories[0]:
        return convert_per_user_history(histories)
    return histories


def convert_per_user_history(users: list, profile_size: int = 80, query_count: int = 5) -> list:
    histories = []
    for user in users:
        examples = user.get("examples", [])
        profile_examples = examples[:profile_size]
        query_examples = examples[profile_size : profile_size + query_count]
        if not query_examples:
            query_examples = examples[:query_count]
        histories.append(
            {
                "user_id": user.get("user_id", ""),
                "source_user": user.get("sender_email", ""),
                "inferred_name": infer_name_from_email(user.get("sender_email", "")),
                "profile": [
                    {
                        "id": item.get("id", ""),
                        "subject": item.get("subject", ""),
                        "body": item.get("reply", ""),
                    }
                    for item in profile_examples
                ],
                "query": [
                    {
                        "id": f"{user.get('user_id', '')}_{item.get('id', index)}",
                        "input": build_query_input(item),
                        "gold": item.get("reply", ""),
                    }
                    for index, item in enumerate(query_examples, start=1)
                ],
            }
        )
    return histories


def build_query_input(item: dict) -> str:
    subject = item.get("subject", "") or "(no subject)"
    received = item.get("received", "")
    return f"Write a reply to this email in the user's style.\n\nSubject: {subject}\n\nIncoming email:\n{received}"


def infer_name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    parts = [part for part in local.replace(".", " ").replace("_", " ").split() if part]
    return " ".join(part.capitalize() for part in parts) or "Unknown"


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
    parser.add_argument(
        "--history",
        default=str(DEFAULT_HISTORY),
        help="History JSON. Defaults to generated data when present, otherwise the checked-in demo file.",
    )
    parser.add_argument(
        "--profiles",
        default=str(DEFAULT_PROFILES),
        help="Profile JSON. Defaults to generated data when present, otherwise the checked-in demo file.",
    )
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--adapter-root", default="")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the app in the default browser after the server starts.",
    )
    args = parser.parse_args()

    history_path = Path(args.history).expanduser()
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history JSON: {history_path}")

    StyleLabHandler.histories = load_web_history(history_path)
    StyleLabHandler.profiles_by_id = load_profiles(Path(args.profiles).expanduser())
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
