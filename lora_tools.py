import argparse
import json
import random
from pathlib import Path
from typing import List

from enron_style import (
    build_generation_messages,
    build_style_fingerprint,
    describe_style_heuristic,
    digit_ratio,
    EmailRecord,
    load_history,
    retrieve_profile_examples,
    tokenize,
)


def export_lora_dataset(
    history_path: Path,
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 7,
    include_profile: bool = True,
    max_profile_rows_per_user: int = 180,
    validation_source: str = "query",
    per_user: bool = False,
) -> dict:
    histories = load_history(history_path)
    rows = []
    rows_by_user = {}
    query_count = 0
    profile_count = 0

    for history in histories:
        user_rows = []
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
                example_max_chars=350,
            )
            messages.append({"role": "assistant", "content": query["gold"]})
            row = {
                "id": query["id"],
                "user_id": history["user_id"],
                "source_user": history.get("source_user", ""),
                "inferred_name": identity,
                "source": "query",
                "messages": messages,
            }
            rows.append(row)
            user_rows.append(row)
            query_count += 1

        if include_profile:
            profile_rows = [
                item for item in profile if is_good_profile_training_example(item.get("body", ""))
            ][:max_profile_rows_per_user]
            for index, item in enumerate(profile_rows, start=1):
                prompt = build_profile_training_prompt(item)
                example_pool = [other for other in profile if other.get("id") != item.get("id")]
                examples = retrieve_profile_examples(example_pool, prompt, top_k=4)
                messages = build_generation_messages(
                    prompt=prompt,
                    style_hint=style_hint,
                    fingerprint=fingerprint,
                    identity=identity,
                    examples=examples,
                    example_max_chars=350,
                )
                messages.append({"role": "assistant", "content": item.get("body", "")})
                row = {
                    "id": f"{history['user_id']}_p{index:04d}",
                    "user_id": history["user_id"],
                    "source_user": history.get("source_user", ""),
                    "inferred_name": identity,
                    "source": "profile",
                    "messages": messages,
                }
                rows.append(row)
                user_rows.append(row)
                profile_count += 1
        rows_by_user[str(history["user_id"])] = user_rows

    rng = random.Random(seed)
    rng.shuffle(rows)
    train_rows, val_rows = split_rows(rows, val_ratio, validation_source)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    users_manifest = {}
    if per_user:
        users_dir = out_dir / "users"
        for user_id, user_rows in rows_by_user.items():
            user_train, user_val = split_rows(user_rows, val_ratio, validation_source)
            user_dir = users_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            user_train_path = user_dir / "train.jsonl"
            user_val_path = user_dir / "val.jsonl"
            write_jsonl(user_train_path, user_train)
            write_jsonl(user_val_path, user_val)
            users_manifest[user_id] = {
                "total": len(user_rows),
                "train": len(user_train),
                "val": len(user_val),
                "train_path": str(user_train_path),
                "val_path": str(user_val_path),
                "recommended_adapter_path": str(Path("data/lora_adapters") / user_id),
            }

    manifest = {
        "total": len(rows),
        "query_rows": query_count,
        "profile_rows": profile_count,
        "train": len(train_rows),
        "val": len(val_rows),
        "validation_source": validation_source,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "users": users_manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def split_rows(rows: List[dict], val_ratio: float, validation_source: str) -> tuple:
    val_candidates = [
        row for row in rows if validation_source == "all" or row.get("source") == validation_source
    ]
    if not val_candidates:
        val_candidates = rows[:]
    val_count = min(
        len(val_candidates),
        max(1, int(round(len(val_candidates) * val_ratio))) if len(val_candidates) > 1 else len(val_candidates),
    )
    val_ids = {row["id"] for row in val_candidates[:val_count]}
    val_rows = [row for row in rows if row["id"] in val_ids]
    train_rows = [row for row in rows if row["id"] not in val_ids]
    if not train_rows and val_rows:
        train_rows = val_rows[:]
    return train_rows, val_rows


def build_profile_training_prompt(item: dict) -> str:
    subject = item.get("subject", "").strip() or "(no subject)"
    return f"Write an email in the user's style for this subject: {subject}"


def is_good_profile_training_example(body: str) -> bool:
    words = tokenize(body)
    if not 5 <= len(words) <= 350:
        return False
    lowered = body.lower()
    blocked = [
        "---------------------- forwarded by",
        "password",
        "login id",
        "auto-generated",
        "unsubscribe",
    ]
    if any(token in lowered for token in blocked):
        return False
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return False
    if digit_ratio(body) > 0.18:
        return False
    return True


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chat-style records for LoRA SFT.")
    parser.add_argument("--history", default="data/processed/user_email_history.json")
    parser.add_argument("--out", default="data/lora")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--no-profile", action="store_true", help="Only export held-out query rows.")
    parser.add_argument("--max-profile-rows-per-user", type=int, default=180)
    parser.add_argument(
        "--per-user",
        action="store_true",
        help="Also export train/val JSONL datasets under OUT/users/<user_id> for per-person adapters.",
    )
    parser.add_argument(
        "--validation-source",
        choices=["query", "profile", "all"],
        default="query",
        help="Which row source to reserve for validation.",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
