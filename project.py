import argparse
import json
import sys
from pathlib import Path

from enron_style import (
    build_dataset,
    evaluate_history,
    generate_style_response,
    load_history,
    maybe_download_kaggle_dataset,
)


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("serve")

    parser = argparse.ArgumentParser(
        description="Build Enron user-style JSON files and generate style-aware replies."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download the Kaggle Enron dataset.")
    download_parser.add_argument(
        "--dataset",
        default="wcukierski/enron-email-dataset",
        help="Kaggle dataset slug.",
    )

    build_parser = subparsers.add_parser("build", help="Process an Enron maildir into JSON schemas.")
    build_parser.add_argument("--maildir", required=True, help="Path to an Enron maildir root.")
    build_parser.add_argument("--out", default="data/processed", help="Output directory.")
    build_parser.add_argument("--prefix", default="t", help="User name prefix filter.")
    build_parser.add_argument("--max-users", type=int, default=100)
    build_parser.add_argument("--profile-size", type=int, default=250)
    build_parser.add_argument("--query-count", type=int, default=5)
    build_parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama/Llama for profile descriptions when available.",
    )

    generate_parser = subparsers.add_parser("generate", help="Generate one response from processed data.")
    generate_parser.add_argument("--history", required=True, help="Path to user_email_history.json.")
    generate_parser.add_argument("--user-id", type=int, help="User id to use; defaults to first user.")
    generate_parser.add_argument("--prompt", required=True, help="Generation task prompt.")
    generate_parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name.")
    generate_parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Use the deterministic fallback instead of local Ollama/Llama.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Generate every query and compare to golds.")
    eval_parser.add_argument("--history", required=True, help="Path to user_email_history.json.")
    eval_parser.add_argument("--limit", type=int, help="Maximum number of queries to evaluate.")
    eval_parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name.")
    eval_parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Use the deterministic fallback instead of local Ollama/Llama.",
    )

    serve_parser = subparsers.add_parser("serve", help="Run the local web interface.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8787)
    serve_parser.add_argument("--history", default="data/processed/user_email_history.json")
    serve_parser.add_argument("--profiles", default="data/processed/profile_user.json")
    serve_parser.add_argument("--model", default="llama3.1:8b")

    lora_parser = subparsers.add_parser("export-lora", help="Export chat SFT records for LoRA training.")
    lora_parser.add_argument("--history", default="data/processed/user_email_history.json")
    lora_parser.add_argument("--out", default="data/lora")
    lora_parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    if args.command == "download":
        path = maybe_download_kaggle_dataset(args.dataset)
        print(path)
        return

    if args.command == "build":
        outputs = build_dataset(
            maildir=Path(args.maildir),
            out_dir=Path(args.out),
            prefix=args.prefix,
            max_users=args.max_users,
            profile_size=args.profile_size,
            query_count=args.query_count,
            use_ollama=args.use_ollama,
        )
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
        return

    if args.command == "generate":
        histories = load_history(Path(args.history))
        selected = histories[0]
        if args.user_id is not None:
            selected = next(item for item in histories if item["user_id"] == args.user_id)
        response = generate_style_response(
            profile=selected["profile"],
            prompt=args.prompt,
            use_ollama=not args.no_ollama,
            model=args.model,
            identity=selected.get("inferred_name", ""),
        )
        print(response)
        return

    if args.command == "evaluate":
        histories = load_history(Path(args.history))
        report = evaluate_history(
            histories=histories,
            limit=args.limit,
            use_ollama=not args.no_ollama,
            model=args.model,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "serve":
        from web_app import main as serve_main

        sys.argv = [
            sys.argv[0],
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--history",
            args.history,
            "--profiles",
            args.profiles,
            "--model",
            args.model,
        ]
        serve_main()
        return

    if args.command == "export-lora":
        from lora_tools import export_lora_dataset

        manifest = export_lora_dataset(
            history_path=Path(args.history),
            out_dir=Path(args.out),
            val_ratio=args.val_ratio,
        )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
