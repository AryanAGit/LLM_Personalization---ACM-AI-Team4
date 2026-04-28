#!/usr/bin/env python3
"""
Process the Enron email dataset into LaMP-style JSON schemas.

Outputs (written to --output-dir):
  - profile_user.json           : 100 items, {id, output}  (user style descriptions)
  - user_email_history.json     : 100 items, {user_id, profile[~100+], query[1+]}
  - user_email_history_label.json : {task, golds: [{id, output}, ...]}

Supported tasks:
  - LaMP_3 : reply generation        (input = previous email, gold = user's reply)
  - LaMP_5 : subject-line generation (input = body,           gold = subject)
  - LaMP_6 : email summarization     (input = body,           gold = 1-sentence summary
                                      produced by an LLM — requires --anthropic-key)

Input:
  - The Kaggle "enron-email-dataset" CSV (emails.csv, columns: file, message), or
  - A maildir directory (the original Enron tarball).

Example:
  python build_enron_lamp.py \\
      --input emails.csv \\
      --output-dir ./out \\
      --task LaMP_5 \\
      --num-users 100 \\
      --profile-size 100 \\
      --queries-per-user 1 \\
      --anthropic-key "$ANTHROPIC_API_KEY"

Dependencies:
  pip install pandas tqdm anthropic
"""
from __future__ import annotations

import argparse
import email
import json
import os
import random
import re
import sys
from collections import defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):  # noqa
        return x

try:
    import anthropic
except ImportError:
    anthropic = None


# ======================================================================
# Email parsing & cleaning
# ======================================================================

FORWARD_MARKERS = (
    "-----Original Message-----",
    "----- Original Message -----",
    "----- Forwarded by",
    "---------------------- Forwarded",
    "________________________________",
)

SIGNATURE_RE = re.compile(r"\n-- ?\n.*", re.DOTALL)
UNDERLINE_SIG_RE = re.compile(r"\n_{5,}\n.*", re.DOTALL)
QUOTED_LINE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)
EMAIL_ADDR_RE = re.compile(r"[\w.\-+]+@[\w.\-]+")
HEADERS_AT_TOP_RE = re.compile(
    r"^(From|To|Sent|Cc|Bcc|Subject|Date):.*$", re.IGNORECASE | re.MULTILINE
)


def _extract_addr(raw: str) -> str:
    m = EMAIL_ADDR_RE.search(raw or "")
    return m.group(0).lower() if m else ""


def parse_message(raw) -> dict:
    """Parse one RFC822 message (bytes or str) → dict of fields + raw body."""
    if isinstance(raw, bytes):
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    else:
        msg = email.message_from_string(raw, policy=policy.default)

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_content()
                    break
                except Exception:
                    continue
    else:
        try:
            body = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body = payload.decode(errors="replace")
            else:
                body = str(payload or "")

    return {
        "from": _extract_addr(msg.get("From") or ""),
        "to": (msg.get("To") or "").strip(),
        "subject": (msg.get("Subject") or "").strip(),
        "date": (msg.get("Date") or "").strip(),
        "in_reply_to": (msg.get("In-Reply-To") or "").strip(),
        "message_id": (msg.get("Message-ID") or "").strip(),
        "body": body or "",
    }


def clean_body(body: str) -> str:
    """Strip signatures, quoted replies, and forwarded sections."""
    if not body:
        return ""
    text = body.replace("\r\n", "\n").replace("\r", "\n")

    # Cut at the earliest forward / original-message marker
    earliest = len(text)
    for marker in FORWARD_MARKERS:
        i = text.find(marker)
        if 0 <= i < earliest:
            earliest = i
    text = text[:earliest]

    text = SIGNATURE_RE.sub("", text)
    text = UNDERLINE_SIG_RE.sub("", text)
    text = QUOTED_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_reply_and_prev(raw_body: str):
    """For LaMP_3: split a reply email into (previous_email, user_reply_text).
    Returns (None, None) if the split cannot be recovered."""
    if not raw_body:
        return None, None
    text = raw_body.replace("\r\n", "\n").replace("\r", "\n")

    split_idx = None
    for marker in FORWARD_MARKERS:
        i = text.find(marker)
        if i > 0:
            split_idx = i if split_idx is None else min(split_idx, i)

    m = re.search(r"\n+On .+ wrote:\s*\n", text)
    if m and (split_idx is None or m.start() < split_idx):
        split_idx = m.start()

    if split_idx is None:
        m = re.search(r"(\n\s*>.*){2,}", text)
        if m and m.start() > 10:
            split_idx = m.start()

    if split_idx is None or split_idx < 5:
        return None, None

    reply_part = text[:split_idx].strip()
    reply_part = SIGNATURE_RE.sub("", reply_part)
    reply_part = UNDERLINE_SIG_RE.sub("", reply_part).strip()

    quoted = text[split_idx:]
    # Strip off the marker line itself
    quoted = re.sub(
        r"^(-+\s*(Original Message|Forwarded by)[^\n]*|_{5,})\n?",
        "",
        quoted,
        flags=re.IGNORECASE,
    )
    # Drop header lines (From:, To:, etc.) that appear before the quoted body
    lines = quoted.splitlines()
    cleaned_lines = []
    past_headers = False
    for line in lines:
        if not past_headers and HEADERS_AT_TOP_RE.match(line):
            continue
        past_headers = True
        cleaned_lines.append(line.lstrip("> ").rstrip())
    quoted = "\n".join(cleaned_lines).strip()

    if len(reply_part) < 20 or len(quoted) < 20:
        return None, None
    return quoted[:4000], reply_part[:4000]


def is_usable(parsed: dict, cleaned: str) -> bool:
    """Heuristic filter for noise, automated, or too-short/long messages."""
    if not cleaned or len(cleaned) < 30 or len(cleaned) > 8000:
        return False
    subj = parsed["subject"].lower()
    frm = parsed["from"]
    if not frm or "@" not in frm:
        return False
    bad_subj = ("out of office", "auto-reply", "undeliverable",
                "delivery failure", "returned mail", "read:")
    if any(t in subj for t in bad_subj):
        return False
    bad_from = ("mailer-daemon", "postmaster", "no-reply", "noreply",
                "announcement", "newsletter")
    if any(t in frm for t in bad_from):
        return False
    if "unsubscribe" in cleaned.lower():
        return False
    return True


# ======================================================================
# Dataset iterators
# ======================================================================

def iter_enron_csv(path: str):
    if pd is None:
        raise RuntimeError("pandas is required for CSV input; `pip install pandas`")
    for chunk in pd.read_csv(path, chunksize=10_000):
        for _, row in chunk.iterrows():
            try:
                yield parse_message(row["message"])
            except Exception:
                continue


def iter_enron_maildir(root: str):
    root_path = Path(root)
    # Prefer sent folders for reliable sender attribution
    sent = [
        p for p in root_path.rglob("*")
        if p.is_file() and any(s in p.parts for s in ("sent", "_sent_mail", "sent_items"))
    ]
    files = sent or [p for p in root_path.rglob("*") if p.is_file()]
    for p in files:
        try:
            with open(p, "rb") as f:
                yield parse_message(f.read())
        except Exception:
            continue


# ======================================================================
# Query (gold-label) construction per task
# ======================================================================

def build_queries_subject(emails, n, rng):
    pool = [
        e for e in emails
        if e["subject"]
        and len(e["subject"].split()) >= 3
        and not e["subject"].lower().startswith(("re:", "fw:", "fwd:"))
        and len(e["body"]) >= 100
    ]
    rng.shuffle(pool)
    out = []
    for e in pool[: n * 5]:
        out.append({
            "input": (
                "Generate a subject line for the following email, "
                "written in the user's style:\n\n" + e["body"]
            ),
            "gold": e["subject"],
            "source_id": e["message_id"],
        })
        if len(out) >= n:
            break
    return out


def build_queries_reply(emails, n, rng):
    pool = []
    for e in emails:
        if not e["subject"].lower().startswith("re:"):
            continue
        prev, reply = extract_reply_and_prev(e["raw_body"])
        if prev and reply:
            pool.append((prev, reply, e["message_id"]))
    rng.shuffle(pool)
    out = []
    for prev, reply, mid in pool[: n * 5]:
        out.append({
            "input": (
                "Write a reply to this email in the user's style:\n\n" + prev
            ),
            "gold": reply,
            "source_id": mid,
        })
        if len(out) >= n:
            break
    return out


def build_queries_summarize(emails, n, rng, client=None, model=None):
    pool = [e for e in emails if len(e["body"]) >= 300]
    rng.shuffle(pool)
    out = []
    for e in pool[: n * 4]:
        if client is not None:
            summary = llm_summarize(client, model, e["body"])
            if not summary:
                continue
        else:
            # Deterministic fallback: trim to first ~30 words of first sentence
            first = re.split(r"(?<=[.!?])\s+", e["body"].strip())[0]
            summary = " ".join(first.split()[:30])
        out.append({
            "input": (
                "Summarize this email in one sentence in the user's style:\n\n"
                + e["body"]
            ),
            "gold": summary,
            "source_id": e["message_id"],
        })
        if len(out) >= n:
            break
    return out


# ======================================================================
# LLM helpers
# ======================================================================

STYLE_PROMPT = """Below are several emails written by a single user. Write a concise (2-4 sentences) description of this user's email writing style and tone. Focus on: greetings/sign-offs, sentence length, formality level, punctuation habits, vocabulary, and any distinctive quirks. Output only the description — no preamble, no labels.

Emails:
{samples}
"""


def describe_user_style(client, model, sample_emails, max_samples=8):
    joined = "\n\n---\n\n".join(
        f"Subject: {e['subject']}\n{e['body']}"
        for e in sample_emails[:max_samples]
    )
    prompt = STYLE_PROMPT.format(samples=joined[:12000])
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def llm_summarize(client, model, body):
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this email in exactly one sentence. "
                    "Output only the sentence — no preamble.\n\n" + body
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception:
        return ""


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Path to emails.csv OR the maildir root directory")
    ap.add_argument("--output-dir", default="./out")
    ap.add_argument("--task", choices=["LaMP_3", "LaMP_5", "LaMP_6"],
                    default="LaMP_5")
    ap.add_argument("--num-users", type=int, default=100)
    ap.add_argument("--profile-size", type=int, default=100,
                    help="Number of profile emails to include per user")
    ap.add_argument("--queries-per-user", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-emails", type=int, default=None,
                    help="Stop parsing after this many raw emails (for quick runs). "
                         "Default: process the whole dataset.")
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="After parsing, cap the number of eligible senders considered "
                         "before user selection. Default: no cap.")
    ap.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    ap.add_argument("--llm-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--skip-style", action="store_true",
                    help="Skip LLM style descriptions (use placeholder text)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    needed_per_user = args.profile_size + args.queries_per_user

    # ---- 1. Parse & group ------------------------------------------------
    print(f"[1/4] Loading Enron data from {args.input} …", file=sys.stderr)
    if os.path.isdir(args.input):
        iterator = iter_enron_maildir(args.input)
    else:
        iterator = iter_enron_csv(args.input)

    by_sender: dict[str, list[dict]] = defaultdict(list)
    parsed_count = 0
    for parsed in tqdm(iterator, desc="parsing", unit="msg"):
        raw_body = parsed["body"]
        cleaned = clean_body(raw_body)
        if not is_usable(parsed, cleaned):
            continue
        by_sender[parsed["from"]].append({
            "message_id": parsed["message_id"]
                          or f"{parsed['from']}-{len(by_sender[parsed['from']])}",
            "subject":    parsed["subject"],
            "body":       cleaned,
            "raw_body":   raw_body,   # needed for LaMP_3 reply extraction
        })
        parsed_count += 1

        # Hard cap on raw emails parsed
        if args.max_emails and parsed_count >= args.max_emails:
            print(f"      reached --max-emails={args.max_emails}, stopping parse",
                  file=sys.stderr)
            break

        # Early exit once we already have enough eligible senders.
        # Check periodically to avoid the O(N) scan on every message.
        if parsed_count % 5000 == 0:
            eligible = sum(1 for v in by_sender.values() if len(v) >= needed_per_user)
            # 3x buffer because some candidates will fail query construction
            if eligible >= args.num_users * 3:
                print(f"      found {eligible} eligible senders after "
                      f"{parsed_count:,} emails, stopping parse early",
                      file=sys.stderr)
                break

    print(f"      parsed {sum(len(v) for v in by_sender.values()):,} usable "
          f"emails from {len(by_sender):,} senders", file=sys.stderr)

    # ---- 2. Pick candidate users ----------------------------------------
    candidates = [(s, m) for s, m in by_sender.items() if len(m) >= needed_per_user]
    rng.shuffle(candidates)
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    print(f"[2/4] {len(candidates):,} senders have ≥ {needed_per_user} usable "
          f"emails (need {args.num_users})", file=sys.stderr)

    # ---- 3. Build per-user records --------------------------------------
    client = None
    if args.anthropic_key and not args.skip_style:
        if anthropic is None:
            print("      WARNING: `anthropic` not installed — style descriptions "
                  "will be placeholders. `pip install anthropic`", file=sys.stderr)
        else:
            client = anthropic.Anthropic(api_key=args.anthropic_key)

    build_fn = {
        "LaMP_3": lambda ems: build_queries_reply(ems, args.queries_per_user, rng),
        "LaMP_5": lambda ems: build_queries_subject(ems, args.queries_per_user, rng),
        "LaMP_6": lambda ems: build_queries_summarize(
            ems, args.queries_per_user, rng,
            client=client, model=args.llm_model,
        ),
    }[args.task]

    print(f"[3/4] Building task={args.task} per-user records …", file=sys.stderr)
    profile_users: list[dict] = []
    user_histories: list[dict] = []
    all_golds: list[dict] = []
    id_mapping: list[dict] = []   # records real sender → anonymized id
    next_user_id = 20_000_001
    next_query_num = 1

    for sender, msgs in tqdm(candidates, desc="users"):
        if len(user_histories) >= args.num_users:
            break

        queries = build_fn(msgs)
        if not queries:
            continue

        # Exclude any email consumed by a query from the user's profile
        used = {q["source_id"] for q in queries}
        profile_pool = [m for m in msgs if m["message_id"] not in used]
        if len(profile_pool) < args.profile_size:
            continue
        rng.shuffle(profile_pool)
        profile_msgs = profile_pool[: args.profile_size]

        uid = next_user_id
        next_user_id += 1
        id_mapping.append({
            "user_id": uid,
            "sender_email": sender,
            "total_emails_available": len(msgs),
        })

        # Style description
        if client is not None:
            try:
                style = describe_user_style(client, args.llm_model, profile_msgs)
            except Exception as exc:
                style = f"(style description unavailable: {exc})"
        else:
            style = ("Placeholder style description. Run with --anthropic-key "
                     "to generate a natural-language style summary.")
        profile_users.append({"id": uid, "output": style})

        history = {
            "user_id": uid,
            "profile": [
                {"id": f"e{i+1:04d}", "subject": m["subject"], "body": m["body"]}
                for i, m in enumerate(profile_msgs)
            ],
            "query": [],
        }
        for q in queries:
            qid = f"q{next_query_num:05d}"
            next_query_num += 1
            history["query"].append({
                "id": qid,
                "input": q["input"],
                "gold":  q["gold"],
            })
            all_golds.append({"id": qid, "output": q["gold"]})
        user_histories.append(history)

    print(f"      finalized {len(user_histories)} users / "
          f"{len(all_golds)} queries", file=sys.stderr)

    if len(user_histories) < args.num_users:
        print(f"      WARNING: only {len(user_histories)} users met all "
              f"constraints (requested {args.num_users}). Lower --profile-size "
              "or loosen filters.", file=sys.stderr)

    # ---- 4. Save ---------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "profile_user.json").write_text(
        json.dumps(profile_users, indent=2, ensure_ascii=False)
    )
    (out / "user_email_history.json").write_text(
        json.dumps(user_histories, indent=2, ensure_ascii=False)
    )
    (out / "user_email_history_label.json").write_text(
        json.dumps({"task": args.task, "golds": all_golds},
                   indent=2, ensure_ascii=False)
    )
    (out / "user_id_mapping.json").write_text(
        json.dumps(id_mapping, indent=2, ensure_ascii=False)
    )
    print(f"[4/4] Wrote outputs to {out}/", file=sys.stderr)
    print(f"      (user_id_mapping.json contains sender↔id mapping — "
          f"keep private, do not ship with training data)", file=sys.stderr)


if __name__ == "__main__":
    main()
