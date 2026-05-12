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

    author_parser = subparsers.add_parser(
        "build-author", help="Convert plain text documents into the shared style-history schema."
    )
    author_parser.add_argument("--input", nargs="+", required=True, help="Plain text corpus files.")
    author_parser.add_argument("--out", default="data/authors/user_email_history.json")
    author_parser.add_argument("--author-id", required=True)
    author_parser.add_argument("--display-name", required=True)
    author_parser.add_argument("--source-name", default="")
    author_parser.add_argument("--profile-size", type=int, default=80)
    author_parser.add_argument("--query-count", type=int, default=12)
    author_parser.add_argument("--min-words", type=int, default=30)
    author_parser.add_argument("--max-words", type=int, default=220)

    generate_parser = subparsers.add_parser("generate", help="Generate one response from processed data.")
    generate_parser.add_argument("--history", required=True, help="Path to user_email_history.json.")
    generate_parser.add_argument("--user-id", help="User id to use; defaults to first user.")
    generate_parser.add_argument("--prompt", required=True, help="Generation task prompt.")
    generate_parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name.")
    generate_parser.add_argument(
        "--backend",
        choices=["fallback", "ollama", "peft"],
        default="ollama",
        help="Generation backend to use.",
    )
    generate_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    generate_parser.add_argument("--adapter-path", default="")
    generate_parser.add_argument("--adapter-root", default="data/lora_adapters")
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
        "--backend",
        choices=["fallback", "ollama", "peft"],
        default="ollama",
        help="Generation backend to use.",
    )
    eval_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    eval_parser.add_argument("--adapter-path", default="")
    eval_parser.add_argument("--adapter-root", default="data/lora_adapters")
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
    serve_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    serve_parser.add_argument("--adapter-path", default="")
    serve_parser.add_argument("--adapter-root", default="data/lora_adapters")

    lora_parser = subparsers.add_parser("export-lora", help="Export chat SFT records for LoRA training.")
    lora_parser.add_argument("--history", default="data/processed/user_email_history.json")
    lora_parser.add_argument("--out", default="data/lora")
    lora_parser.add_argument("--val-ratio", type=float, default=0.2)
    lora_parser.add_argument("--no-profile", action="store_true", help="Only export held-out query rows.")
    lora_parser.add_argument("--max-profile-rows-per-user", type=int, default=180)
    lora_parser.add_argument(
        "--per-user",
        action="store_true",
        help="Also export train/val JSONL datasets under OUT/users/<user_id> for per-person adapters.",
    )
    lora_parser.add_argument(
        "--validation-source",
        choices=["query", "profile", "all"],
        default="query",
        help="Which row source to reserve for validation.",
    )

    train_user_lora_parser = subparsers.add_parser(
        "train-user-lora", help="Train one per-user LoRA adapter from exported per-user JSONL."
    )
    train_user_lora_parser.add_argument("--base-model", required=True)
    train_user_lora_parser.add_argument("--lora-dir", default="data/lora")
    train_user_lora_parser.add_argument("--user-id", required=True)
    train_user_lora_parser.add_argument("--adapter-root", default="data/lora_adapters")
    train_user_lora_parser.add_argument("--cache-dir", default="data/hf_cache")
    train_user_lora_parser.add_argument("--max-steps", type=int, default=100)
    train_user_lora_parser.add_argument("--max-length", type=int, default=1536)
    train_user_lora_parser.add_argument("--lr", type=float, default=2e-4)
    train_user_lora_parser.add_argument("--target-modules", default="all-linear")
    train_user_lora_parser.add_argument("--lora-r", type=int, default=16)
    train_user_lora_parser.add_argument("--lora-alpha", type=int, default=32)
    train_user_lora_parser.add_argument("--no-eval", action="store_true")

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

    if args.command == "build-author":
        from style_corpus import build_author_history, load_text_documents, write_author_history

        history = build_author_history(
            documents=load_text_documents(Path(item) for item in args.input),
            author_id=args.author_id,
            display_name=args.display_name,
            source_name=args.source_name,
            profile_size=args.profile_size,
            query_count=args.query_count,
            min_words=args.min_words,
            max_words=args.max_words,
        )
        write_author_history(history, Path(args.out))
        print(
            json.dumps(
                {"out": args.out, "profile": len(history["profile"]), "query": len(history["query"])},
                indent=2,
            )
        )
        return

    if args.command == "generate":
        histories = load_history(Path(args.history))
        selected = histories[0]
        if args.user_id is not None:
            selected = next(item for item in histories if str(item["user_id"]) == str(args.user_id))
        response = generate_style_response(
            profile=selected["profile"],
            prompt=args.prompt,
            use_ollama=not args.no_ollama,
            model=args.model,
            backend="fallback" if args.no_ollama else args.backend,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            adapter_root=args.adapter_root,
            user_id=selected.get("user_id", ""),
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
            backend="fallback" if args.no_ollama else args.backend,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            adapter_root=args.adapter_root,
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
            "--base-model",
            args.base_model,
            "--adapter-path",
            args.adapter_path,
            "--adapter-root",
            args.adapter_root,
        ]
        serve_main()
        return

    if args.command == "export-lora":
        from lora_tools import export_lora_dataset

        manifest = export_lora_dataset(
            history_path=Path(args.history),
            out_dir=Path(args.out),
            val_ratio=args.val_ratio,
            include_profile=not args.no_profile,
            max_profile_rows_per_user=args.max_profile_rows_per_user,
            validation_source=args.validation_source,
            per_user=args.per_user,
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "train-user-lora":
        from train_lora import main as train_lora_main

        user_dir = Path(args.lora_dir) / "users" / str(args.user_id)
        train_path = user_dir / "train.jsonl"
        val_path = user_dir / "val.jsonl"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Missing per-user LoRA files under {user_dir}. Run export-lora --per-user first."
            )

        sys.argv = [
            sys.argv[0],
            "--base-model",
            args.base_model,
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--out",
            str(Path(args.adapter_root) / str(args.user_id)),
            "--cache-dir",
            args.cache_dir,
            "--max-steps",
            str(args.max_steps),
            "--max-length",
            str(args.max_length),
            "--lr",
            str(args.lr),
            "--target-modules",
            args.target_modules,
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
        ]
        if args.no_eval:
            sys.argv.append("--no-eval")
        train_lora_main()
        return


if __name__ == "__main__":
    main()
