import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence


def build_author_history(
    documents: Sequence[dict],
    author_id: str,
    display_name: str,
    source_name: str = "",
    profile_size: int = 80,
    query_count: int = 12,
    min_words: int = 30,
    max_words: int = 220,
) -> dict:
    passages = []
    for document in documents:
        title = document.get("title", "")
        for index, passage in enumerate(
            split_passages(document.get("text", ""), min_words=min_words, max_words=max_words),
            start=1,
        ):
            passage_title, passage_body = extract_passage_title(passage)
            passages.append(
                {
                    "id": f"{author_id}_p{len(passages) + 1:04d}",
                    "subject": passage_title or infer_passage_topic(passage_body) or title or f"Passage {index}",
                    "body": passage_body,
                }
            )

    needed = profile_size + query_count
    if len(passages) < max(2, needed):
        raise ValueError(
            f"Not enough usable passages for {display_name}: found {len(passages)}, need at least {needed}."
        )

    profile = passages[:profile_size]
    query_passages = passages[profile_size : profile_size + query_count]
    queries = [
        {
            "id": f"{author_id}_q{index:03d}",
            "input": build_author_prompt(display_name, item["subject"]),
            "gold": item["body"],
        }
        for index, item in enumerate(query_passages, start=1)
    ]

    return {
        "user_id": author_id,
        "source_user": source_name or display_name,
        "inferred_name": display_name,
        "profile": profile,
        "query": queries,
    }


def split_passages(text: str, min_words: int = 30, max_words: int = 220) -> List[str]:
    cleaned = normalize_corpus_text(text)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    passages = []
    buffer = []
    buffer_words = 0

    for paragraph in paragraphs:
        words = re.findall(r"\w+", paragraph)
        if not words:
            continue
        if len(words) > max_words:
            for sentence_group in split_long_paragraph(paragraph, min_words, max_words):
                passages.append(sentence_group)
            continue
        if buffer and buffer_words + len(words) > max_words:
            maybe_add_passage(passages, " ".join(buffer), min_words)
            buffer = []
            buffer_words = 0
        buffer.append(paragraph)
        buffer_words += len(words)
        if buffer_words >= min_words:
            maybe_add_passage(passages, "\n\n".join(buffer), min_words)
            buffer = []
            buffer_words = 0

    if buffer:
        maybe_add_passage(passages, "\n\n".join(buffer), min_words)
    return passages


def split_long_paragraph(paragraph: str, min_words: int, max_words: int) -> List[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
    passages = []
    buffer = []
    buffer_words = 0
    for sentence in sentences:
        count = len(re.findall(r"\w+", sentence))
        if buffer and buffer_words + count > max_words:
            maybe_add_passage(passages, " ".join(buffer), min_words)
            buffer = []
            buffer_words = 0
        buffer.append(sentence)
        buffer_words += count
    if buffer:
        maybe_add_passage(passages, " ".join(buffer), min_words)
    return passages


def maybe_add_passage(passages: List[str], text: str, min_words: int) -> None:
    if len(re.findall(r"\w+", text)) >= min_words:
        passages.append(text.strip())


def extract_passage_title(passage: str) -> tuple:
    lines = [line.strip() for line in passage.splitlines() if line.strip()]
    if len(lines) < 2:
        return "", passage
    if lines[0].startswith("_"):
        heading_lines = []
        body_start = 0
        for index, line in enumerate(lines):
            heading_lines.append(line)
            if line.endswith("_"):
                body_start = index + 1
                break
        if body_start:
            normalized = " ".join(heading_lines).strip("_* ")
            body = "\n\n".join(lines[body_start:]).strip()
            if len(re.findall(r"\w+", body)) >= 10:
                return normalized, body
    first = lines[0]
    normalized = first.strip("_* ")
    looks_like_heading = (
        first.startswith("_")
        or first.isupper()
        or (len(normalized.split()) <= 16 and normalized.endswith((")", "1865", "1864", "1863", "1862", "1861")))
    )
    if not looks_like_heading:
        return "", passage
    body = "\n\n".join(lines[1:]).strip()
    if len(re.findall(r"\w+", body)) < 10:
        return "", passage
    return normalized, body


def infer_passage_topic(passage: str, max_words: int = 12) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", passage)
    if not words:
        return ""
    topic = " ".join(words[:max_words])
    return topic[0].upper() + topic[1:] if topic else ""


def normalize_corpus_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = strip_known_editorial_front_matter(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
        r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*",
    ]
    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            text = text[match.end() :]
            break
    for pattern in end_patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            text = text[: match.start()]
            break
    return text


def strip_known_editorial_front_matter(text: str) -> str:
    headings = [
        r"(?m)^LINCOLN'S SPEECHES AND LETTERS\s*$",
    ]
    for heading in headings:
        match = re.search(heading, text)
        if match:
            return text[match.end() :]
    return text


def build_author_prompt(display_name: str, subject: str) -> str:
    subject_text = subject.strip() or "a new topic"
    return (
        f"Write a new passage in {display_name}'s style inspired by this topic: {subject_text}. "
        "Do not quote or continue an existing source passage."
    )


def load_text_documents(paths: Iterable[Path]) -> List[dict]:
    documents = []
    for path in paths:
        documents.append(
            {
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def write_author_history(history: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([history], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a style history JSON from plain text files.")
    parser.add_argument("--input", nargs="+", required=True, help="Plain text corpus files.")
    parser.add_argument("--out", default="data/authors/user_email_history.json")
    parser.add_argument("--author-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--source-name", default="")
    parser.add_argument("--profile-size", type=int, default=80)
    parser.add_argument("--query-count", type=int, default=12)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--max-words", type=int, default=220)
    args = parser.parse_args()

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
    print(json.dumps({"out": args.out, "profile": len(history["profile"]), "query": len(history["query"])}, indent=2))


if __name__ == "__main__":
    main()
