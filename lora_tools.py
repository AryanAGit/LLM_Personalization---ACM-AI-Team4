import argparse
import json
import random
from pathlib import Path
from typing import List

from enron_style import (
    build_generation_messages,
    build_style_fingerprint,
    describe_style_heuristic,
    EmailRecord,
    load_history,
    retrieve_profile_examples,
)


def export_lora_dataset(
    history_path: Path,
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 7,
) -> dict:
    histories = load_history(history_path)
    rows = []

    for history in histories:
        profile = history["profile"]
        identity = history.get("inferred_name", "")
        style_hint = describe_style_heuristic(
            [
                EmailRecord(
                    id=item["id"],
                    user_name=str(history["user_id"]),
                    subject=item.get("subject", ""),
                    body=item.get("body", ""),
                )
                for item in profile
            ]
        )
        fingerprint = build_style_fingerprint(profile)

        for query in history["query"]:
            examples = retrieve_profile_examples(profile, query["input"], top_k=8)
            messages = build_generation_messages(
                prompt=query["input"],
                style_hint=style_hint,
                fingerprint=fingerprint,
                identity=identity,
                examples=examples,
            )
            messages.append({"role": "assistant", "content": query["gold"]})
            rows.append(
                {
                    "id": query["id"],
                    "user_id": history["user_id"],
                    "source_user": history.get("source_user", ""),
                    "inferred_name": identity,
                    "messages": messages,
                }
            )

    random.Random(seed).shuffle(rows)
    split_at = max(1, int(len(rows) * (1 - val_ratio))) if len(rows) > 1 else len(rows)
    train_rows = rows[:split_at]
    val_rows = rows[split_at:] or rows[-1:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chat-style records for LoRA SFT.")
    parser.add_argument("--history", default="data/processed/user_email_history.json")
    parser.add_argument("--out", default="data/lora")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    manifest = export_lora_dataset(
        history_path=Path(args.history),
        out_dir=Path(args.out),
        val_ratio=args.val_ratio,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
