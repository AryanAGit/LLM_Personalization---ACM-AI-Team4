import csv
import json
import re
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SENT_DIR_NAMES = {"sent", "sent_items", "_sent_mail"}


@dataclass(frozen=True)
class EmailRecord:
    id: str
    user_name: str
    subject: str
    body: str
    context: str = ""


def maybe_download_kaggle_dataset(dataset: str = "wcukierski/enron-email-dataset") -> str:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install it and configure Kaggle API credentials first."
        ) from exc
    return kagglehub.dataset_download(dataset)


def load_history(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_dataset(
    maildir: Path,
    out_dir: Path,
    prefix: str = "t",
    max_users: int = 100,
    profile_size: int = 100,
    query_count: int = 5,
    use_ollama: bool = False,
    model: str = "llama3.1:8b",
) -> Dict[str, Path]:
    if not maildir.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {maildir}. Replace the example path with the real "
            "unzipped Kaggle folder or emails.csv path."
        )

    records = list(read_sent_emails(maildir))
    if not records:
        raise ValueError(
            f"No sent emails were found in {maildir}. Expected either Kaggle emails.csv or an "
            "Enron maildir with sent, sent_items, or _sent_mail folders."
        )

    grouped = group_by_user(records)
    min_emails = min(profile_size + 1, max(query_count + 20, query_count + 1))
    selected = select_users(grouped, prefix=prefix, max_users=max_users, min_emails=min_emails)

    if not selected:
        raise ValueError(
            f"No users starting with {prefix!r} had at least {min_emails} sent emails."
        )

    profile_users = []
    histories = []
    golds = []

    for index, user_name in enumerate(selected):
        user_id = 20_000_001 + index
        emails = grouped[user_name]
        usable_profile_size = min(profile_size, max(len(emails) - query_count, 1))
        profile_records = emails[:usable_profile_size]
        query_pool = emails[usable_profile_size:]
        query_records = select_query_records(query_pool, query_count)
        if not query_records:
            query_records = emails[-1:]
            profile_records = emails[:-1]

        profile = [email_to_schema(record) for record in profile_records]
        user_name_from_profile = infer_profile_identity(profile)
        queries = []
        for query_index, record in enumerate(query_records, start=1):
            query_id = f"{user_id}_q{query_index:03d}"
            task_prompt = (
                build_query_prompt(record)
            )
            queries.append({"id": query_id, "input": task_prompt, "gold": record.body})
            golds.append({"id": query_id, "output": record.body})

        style = describe_style(profile_records, use_ollama=use_ollama, model=model)
        profile_users.append({"id": user_id, "output": style})
        histories.append(
            {
                "user_id": user_id,
                "source_user": user_name,
                "inferred_name": user_name_from_profile,
                "profile": profile,
                "query": queries,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "profile_user": out_dir / "profile_user.json",
        "user_email_history": out_dir / "user_email_history.json",
        "user_email_history_label": out_dir / "user_email_history_label.json",
    }
    write_json(outputs["profile_user"], profile_users)
    write_json(outputs["user_email_history"], histories)
    write_json(outputs["user_email_history_label"], {"task": "LaMP_3", "golds": golds})
    return outputs


def read_sent_emails(maildir: Path) -> Iterable[EmailRecord]:
    csv_path = _find_emails_csv(maildir)
    if csv_path:
        yield from read_sent_emails_csv(csv_path)
        return

    for path in sorted(maildir.rglob("*")):
        if not path.is_file() or not _looks_like_sent_path(path, maildir):
            continue
        record = parse_email(path, maildir)
        if record and record.body:
            yield record


def read_sent_emails_csv(csv_path: Path) -> Iterable[EmailRecord]:
    csv.field_size_limit(sys.maxsize)
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row.get("file", "")
            if not _looks_like_sent_parts(Path(file_name).parts):
                continue
            raw_message = row.get("message", "")
            record = parse_email_text(raw_message, file_name)
            if record and record.body:
                yield record


def parse_email(path: Path, maildir: Path) -> Optional[EmailRecord]:
    try:
        with path.open("rb") as handle:
            message = BytesParser(policy=policy.default).parse(handle)
    except (OSError, UnicodeError):
        return None

    user_name = path.relative_to(maildir).parts[0].lower()
    return message_to_record(message, user_name, path.name)


def parse_email_text(raw_message: str, file_name: str) -> Optional[EmailRecord]:
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_message.encode("utf-8", errors="ignore"))
    except (OSError, UnicodeError):
        return None

    parts = Path(file_name).parts
    if not parts:
        return None
    user_name = parts[0].lower()
    return message_to_record(message, user_name, file_name)


def message_to_record(message, user_name: str, fallback_id: str) -> Optional[EmailRecord]:
    raw_body = extract_text_body(message)
    body = normalize_body(raw_body)
    if not body:
        return None
    context = extract_reply_context(raw_body)
    subject = normalize_subject(str(message.get("subject", "")))
    message_id = str(message.get("message-id", "")).strip("<> ")
    email_id = message_id or f"{user_name}/{fallback_id}"
    return EmailRecord(id=email_id, user_name=user_name, subject=subject, body=body, context=context)


def extract_text_body(message) -> str:
    if message.is_multipart():
        parts = []
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                parts.append(_safe_get_content(part))
        return "\n".join(parts)
    return _safe_get_content(message)


def normalize_body(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"\n\s*-{2,}\s*Original Message\s*-{2,}.*", "", body, flags=re.I | re.S)
    body = re.sub(r"\n\s*-{2,}\s*Forwarded by .*", "", body, flags=re.I | re.S)
    body = re.sub(r"\nFrom: .*", "", body, flags=re.I | re.S)
    body = re.sub(r"\n[ \t]*>.*", "", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def extract_reply_context(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    patterns = [
        r"\n\s*-{2,}\s*Original Message\s*-{2,}\s*(.*)",
        r"\n\s*-{2,}\s*Forwarded by\s+.*",
        r"\nFrom:\s+.*",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.I | re.S)
        if match:
            context = match.group(1) if match.lastindex else body[match.start() :]
            return trim_text(clean_context(context), 2000)
    return ""


def clean_context(context: str) -> str:
    context = re.sub(r"\n[ \t]*> ?", "\n", context)
    context = re.sub(r"[ \t]+\n", "\n", context)
    context = re.sub(r"\n{3,}", "\n\n", context)
    return context.strip()


def normalize_subject(subject: str) -> str:
    return re.sub(r"\s+", " ", subject).strip()


def group_by_user(records: Iterable[EmailRecord]) -> Dict[str, List[EmailRecord]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.user_name].append(record)
    return {user: emails for user, emails in grouped.items()}


def select_users(
    grouped: Dict[str, List[EmailRecord]], prefix: str = "t", max_users: int = 100, min_emails: int = 101
) -> List[str]:
    candidates = [
        (user, len(emails))
        for user, emails in grouped.items()
        if user.lower().startswith(prefix.lower()) and len(emails) >= min_emails
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [user for user, _count in candidates[:max_users]]


def select_query_records(records: Sequence[EmailRecord], query_count: int) -> List[EmailRecord]:
    context_records = [record for record in records if record.context]
    plain_records = [record for record in records if not record.context]
    return list(context_records[:query_count]) + list(plain_records[: max(query_count - len(context_records), 0)])


def describe_style(
    emails: Sequence[EmailRecord], use_ollama: bool = False, model: str = "llama3.1:8b"
) -> str:
    heuristic = describe_style_heuristic(emails)
    if not use_ollama:
        return heuristic

    samples = "\n\n".join(email.body[:1200] for email in emails[:8])
    prompt = (
        "Describe this user's email writing style and tone in one concise paragraph. "
        "Mention greeting habits, paragraph length, tone, sign-offs, and punctuation.\n\n"
        f"{samples}"
    )
    try:
        from ollama import chat

        response = chat(model=model, messages=[{"role": "user", "content": prompt}])
        content = getattr(getattr(response, "message", None), "content", None)
        return str(content or heuristic).strip()
    except Exception:
        return heuristic


def describe_style_heuristic(emails: Sequence[EmailRecord]) -> str:
    bodies = [email.body for email in emails if email.body]
    words = [len(re.findall(r"\w+", body)) for body in bodies]
    avg_words = int(sum(words) / max(len(words), 1))
    greeting_rate = sum(1 for body in bodies if re.match(r"^(hi|hello|hey|dear)\b", body, re.I))
    thanks_rate = sum(1 for body in bodies if re.search(r"\b(thanks|thank you|regards)\b", body, re.I))
    exclamation_rate = sum(1 for body in bodies if "!" in body)
    paragraphs = [body.count("\n\n") + 1 for body in bodies]
    avg_paragraphs = sum(paragraphs) / max(len(paragraphs), 1)

    length = "brief" if avg_words < 75 else "moderately detailed" if avg_words < 180 else "detailed"
    tone = "direct and practical"
    if thanks_rate / max(len(bodies), 1) > 0.35:
        tone = "courteous and practical"
    if exclamation_rate / max(len(bodies), 1) > 0.2:
        tone = "warm and energetic"
    greeting = "often opens with a greeting" if greeting_rate / max(len(bodies), 1) > 0.35 else "usually gets straight to the point"
    signoff = "frequently closes with thanks or a polite sign-off" if thanks_rate / max(len(bodies), 1) > 0.35 else "uses minimal sign-offs"
    paragraph_style = "short paragraphs" if avg_paragraphs >= 2 else "compact single-block messages"
    return (
        f"The user's emails are {length}, {tone}. They {greeting}, write in "
        f"{paragraph_style}, and {signoff}. Their messages average about {avg_words} words."
    )


def author_style_summary(summary: str) -> str:
    return (
        summary.replace("The user's emails are", "The source passages are")
        .replace("They usually gets", "They usually get")
        .replace("and uses", "and use")
        .replace("Their messages average", "The passages average")
        .replace("greeting", "opening")
        .replace("sign-offs", "closings")
    )


def generate_style_response(
    profile: Sequence[dict],
    prompt: str,
    use_ollama: bool = False,
    model: str = "llama3.1:8b",
    top_k: int = 8,
    identity: str = "",
    backend: str = "",
    adapter_path: str = "",
    adapter_root: str = "",
    base_model: str = "",
    user_id=0,
    rag_mode: str = "enhanced",
) -> str:
    backend = normalize_backend(backend, use_ollama)
    if backend == "peft" and is_freeform_email_instruction(prompt):
        top_k = 0
    identity = identity or infer_profile_identity(profile)
    author_mode = is_author_style_prompt(prompt) or should_use_author_style(prompt, identity)
    retrieval_prompt = extract_author_topic(prompt) if author_mode else prompt
    retrieved = retrieve_profile_examples(profile, retrieval_prompt, top_k=top_k, mode=rag_mode)
    style_hint = describe_style_heuristic(
        [
            EmailRecord(
                id=item["id"],
                user_name="profile",
                subject=item.get("subject", ""),
                body=item.get("body", ""),
            )
            for item in profile
        ]
    )
    fingerprint = build_style_fingerprint(profile)
    if author_mode:
        style_hint = author_style_summary(style_hint)
    messages = build_generation_messages(
        prompt=prompt,
        style_hint=style_hint,
        fingerprint=fingerprint,
        identity=identity,
        examples=retrieved,
    )

    if backend == "ollama":
        try:
            from ollama import chat

            response = chat(
                model=model,
                messages=messages,
                options={
                    "temperature": 0.45 if author_mode else 0.2,
                    "top_p": 0.9,
                    "num_predict": 260 if author_mode else 180,
                    "repeat_penalty": 1.08,
                },
            )
            content = getattr(getattr(response, "message", None), "content", None)
            if content:
                cleaned = clean_model_response(str(content), identity=identity, prompt=prompt)
                if author_mode and looks_like_failed_generation(cleaned):
                    raise RuntimeError("Ollama returned unusable historical-style text.")
                if author_mode and has_source_copy_risk(cleaned, profile):
                    rewrite_messages = messages + [
                        {"role": "assistant", "content": cleaned},
                        {
                            "role": "user",
                            "content": (
                                "Rewrite the passage for the same task in the same historical style, but do not reuse "
                                "any exact sentence or distinctive phrase from the source examples. Preserve the style, "
                                "cadence, and argument structure while using new wording. Write only the revised passage."
                            ),
                        },
                    ]
                    response = chat(
                        model=model,
                        messages=rewrite_messages,
                        options={
                            "temperature": 0.55,
                            "top_p": 0.9,
                            "num_predict": 260,
                            "repeat_penalty": 1.12,
                        },
                    )
                    content = getattr(getattr(response, "message", None), "content", None)
                    if content:
                        cleaned = clean_model_response(str(content), identity=identity, prompt=prompt)
                return cleaned
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    if backend == "peft":
        try:
            from lora_backend import generate_with_peft, resolve_adapter_path

            resolved_adapter = resolve_adapter_path(adapter_path, adapter_root, user_id)
            if not resolved_adapter and (adapter_path or adapter_root):
                raise FileNotFoundError(
                    f"No LoRA adapter found for user_id {user_id}. Train one with project.py train-user-lora "
                    f"or provide --adapter-path explicitly."
                )
            content = generate_with_peft(
                messages=messages,
                base_model=base_model or model,
                adapter_path=resolved_adapter,
            )
            cleaned = clean_model_response(content, identity=identity, prompt=prompt)
            cleaned = enforce_outbound_recipient(cleaned, prompt)
            if looks_like_failed_generation(cleaned):
                raise RuntimeError(
                    "The LoRA model returned unusable text. This usually means the adapter is undertrained, "
                    "trained on the wrong data, or paired with the wrong base model."
                )
            return cleaned
        except Exception as exc:
            raise RuntimeError(f"PEFT/LoRA generation failed: {exc}") from exc

    return heuristic_response(prompt, retrieved, style_hint, identity=identity)


def normalize_backend(backend: str, use_ollama: bool) -> str:
    normalized = (backend or "").strip().lower()
    if normalized in {"fallback", "heuristic", "local"}:
        return "fallback"
    if normalized in {"ollama", "llama"}:
        return "ollama"
    if normalized in {"peft", "lora", "hf", "huggingface", "transformers"}:
        return "peft"
    return "ollama" if use_ollama else "fallback"


def build_generation_messages(
    prompt: str,
    style_hint: str,
    fingerprint: dict,
    identity: str,
    examples: Sequence[dict],
    example_max_chars: int = 900,
) -> List[dict]:
    example_text = "\n\n".join(
        f"Example {index}\nSubject: {item.get('subject', '')}\nBody:\n{trim_text(item.get('body', ''), example_max_chars)}"
        for index, item in enumerate(examples, start=1)
    )
    identity_text = identity or "unknown"
    intent_guidance = infer_intent_guidance(prompt, identity)
    author_mode = is_author_style_prompt(prompt) or should_use_author_style(prompt, identity)
    if author_mode:
        system = (
            f"You write original public prose in the historical style of {identity_text}. "
            "Use the examples only to infer cadence, sentence length, diction, moral framing, transitions, and rhetorical structure. "
            "Do not quote, paraphrase closely, or continue any source passage. Do not include modern assistant phrases. "
            "Answer the new prompt directly while preserving the target voice. "
            "For Abraham Lincoln, prefer measured public argument: plain opening claim, moral principle, practical consequence, "
            "balanced contrasts, words such as duty, justice, free, people, law, labor, patience, and union when appropriate. "
            "Do not use salutations such as 'Dear friends' or 'My dear colleagues' unless the prompt explicitly asks for a letter. "
            "Do not sound like a modern office memo; translate practical requests into sober public exhortation. "
            "Write 130 to 230 words unless the user asks otherwise. "
            "Never include a preamble, analysis, title, or Subject line. Write only the passage."
        )
    else:
        system = (
            "You write email replies as one specific person. Infer style from the examples: "
            "length, directness, greeting habits, paragraphing, punctuation, and sign-off habits. "
            "If the task asks you to write an outbound email to someone, draft that outbound email only. "
            "Do not answer as a helper, do not say what files you have, and do not invent attachments or lists. "
            "Do not copy facts, names, apologies, or events from examples unless the task itself provides them. "
            "Answer the actual task intent. If the task is a direct question to the user, answer directly in first person. "
            "Write from the inferred user's perspective; do not repeat claims about that user as if they came from someone else. "
            f"The only allowed signature name is {identity or 'the inferred user'}; never sign as another person from the examples. "
            "If the incoming email asks for information or files and the evidence is unclear, be cautious: say what you do not have, "
            "suggest who to check with only when supported by the incoming email or similar examples, and do not invent ownership. "
            "If the task says 'reply to NAME:' or 'write a reply to NAME:', treat the text after the colon as "
            "the incoming message from NAME and respond to it. "
            "Do not claim you have completed an action unless the task says it has already been completed. "
            "Never include a preamble, analysis, 'here it is', or a Subject line. "
            "Write only the requested text, with no explanation."
        )
    user = (
        f"Inferred user identity: {identity_text}\n"
        f"Style summary: {style_hint}\n\n"
        f"Style fingerprint: {json.dumps(fingerprint, ensure_ascii=False)}\n\n"
        f"Intent guidance: {intent_guidance}\n\n"
        f"Similar source examples for style only:\n{example_text}\n\n"
        f"Task:\n{prompt}\n\n"
        "Response:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def infer_intent_guidance(prompt: str, identity: str = "") -> str:
    lower_prompt = prompt.lower()
    if is_introduction_prompt(prompt):
        if identity:
            return (
                f"The sender is introducing themselves and asking for your name. Reply naturally, "
                f"acknowledge their name if provided, and introduce yourself as {identity}."
            )
        return "The sender is introducing themselves and asking for your name. Reply naturally without inventing a full identity."
    if is_name_question_prompt(prompt):
        if identity:
            return f"Answer the name question directly as {identity}."
        return "Answer the name question directly, without inventing a name."
    if re.search(r"\bhow are you\b|\bhow are you doing\b", lower_prompt):
        return "Answer the social question briefly and naturally in first person."
    if re.search(r"\bplease review\b|\breview the attached\b|\breview this\b", lower_prompt):
        return "The sender is asking for a review. Say you will review or take a look; do not say you already reviewed it."
    if re.search(r"\bwrite (?:me )?an email to\b.*\bfiles?\b|\bfiles?\b.*\b(?:wednesday|monday|tuesday|thursday|friday|deadline|due)\b|\bneed\b.*\bfiles?\b", lower_prompt):
        return (
            "The task is to draft an outbound email about a deadline for files. Write to the named recipient, "
            "state that the files need to be in by the requested day, and do not list or invent file contents."
        )
    if re.search(r"\bdue diligence\b|\bphysical bandwidth\b|\bhelp you find\b|\bforward\b", lower_prompt):
        return (
            "The sender is asking for information or files. If the incoming thread does not prove the user has them, "
            "reply cautiously in the user's voice, say you do not have or are not sure, and redirect only to contacts "
            "supported by the incoming thread. Do not claim the work was performed under your direction just because "
            "someone else said that in the thread."
        )
    if re.search(r"\breschedule\b|\bmove (?:the )?(?:call|meeting)\b", lower_prompt):
        return "The sender is asking about scheduling. Reply with availability or agreement; do not invent unrelated details."
    return "Infer the requested reply intent from the task and answer it directly."


def is_introduction_prompt(prompt: str) -> bool:
    asks_name = is_name_question_prompt(prompt)
    gives_name = bool(re.search(r"\bmy name(?: is|'s|s)\b|\bi'?m\s+[A-Z]?", prompt, re.I))
    return asks_name and gives_name


def is_name_question_prompt(prompt: str) -> bool:
    return bool(
        re.search(
            r"\bwhat(?:'?s| is)? your name\b|\bwhat your name\b|\bwho are you\b",
            prompt.lower(),
        )
    )


def is_freeform_email_instruction(prompt: str) -> bool:
    lower = prompt.lower()
    if "incoming email:" in lower or "subject:" in lower:
        return False
    return bool(re.search(r"\bwrite (?:me )?an email to\b|\btelling (?:him|her|them)\b", lower))


def is_author_style_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    return (
        "new passage" in lower
        or "inspired by this topic" in lower
        or "public statement" in lower
        or "write a paragraph" in lower
    )


def should_use_author_style(prompt: str, identity: str = "") -> bool:
    lower = prompt.lower()
    if not identity:
        return False
    if is_freeform_email_instruction(prompt) or "incoming email:" in lower:
        return False
    if re.search(r"\breply\b|\bemail\b|\bwhat'?s your name\b|\bhow are you\b", lower):
        return False
    return bool(re.search(r"\bwrite\b|\bexplain\b|\bstatement\b|\bappeal\b|\bnote\b", lower))


def enforce_outbound_recipient(text: str, prompt: str) -> str:
    if not is_freeform_email_instruction(prompt):
        return text
    recipient = extract_prompt_recipient(prompt)
    if not recipient:
        return text
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if recipient.lower() in first_line.lower():
        return text
    return f"{recipient},\n\n{text.strip()}"


def evaluate_history(
    histories: Sequence[dict],
    limit: Optional[int] = None,
    use_ollama: bool = False,
    model: str = "llama3.1:8b",
    backend: str = "",
    adapter_path: str = "",
    adapter_root: str = "",
    base_model: str = "",
) -> dict:
    examples = []
    scores = []
    count = 0

    for history in histories:
        user_id = history["user_id"]
        profile = history["profile"]
        for query in history["query"]:
            if limit is not None and count >= limit:
                return summarize_evaluation(scores, examples)

            prediction = generate_style_response(
                profile=profile,
                prompt=query["input"],
                use_ollama=use_ollama,
                model=model,
                backend=backend,
                adapter_path=adapter_path,
                adapter_root=adapter_root,
                base_model=base_model,
                user_id=history.get("user_id", ""),
                identity=history.get("inferred_name", ""),
            )
            score = score_prediction(prediction, query["gold"])
            score.update(score_against_profile(prediction, profile))
            score.update({"user_id": user_id, "query_id": query["id"]})
            scores.append(score)
            if len(examples) < 3:
                examples.append(
                    {
                        "user_id": user_id,
                        "query_id": query["id"],
                        "input": query["input"],
                        "prediction": prediction,
                        "gold": query["gold"],
                        "scores": score,
                    }
                )
            count += 1

    return summarize_evaluation(scores, examples)


def summarize_evaluation(scores: Sequence[dict], examples: Sequence[dict]) -> dict:
    if not scores:
        return {"count": 0, "metrics": {}, "examples": []}

    metric_names = [
        "word_f1",
        "length_ratio",
        "greeting_match",
        "signoff_match",
        "profile_copy_5gram_rate",
        "profile_longest_copy_run",
        "style_distance",
    ]
    metrics = {
        name: round(sum(score[name] for score in scores) / len(scores), 4)
        for name in metric_names
    }
    return {"count": len(scores), "metrics": metrics, "examples": list(examples)}


def score_prediction(prediction: str, gold: str) -> dict:
    prediction_terms = tokenize(prediction)
    gold_terms = tokenize(gold)
    prediction_counts = Counter(prediction_terms)
    gold_counts = Counter(gold_terms)
    overlap = sum((prediction_counts & gold_counts).values())
    precision = overlap / max(len(prediction_terms), 1)
    recall = overlap / max(len(gold_terms), 1)
    word_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    prediction_len = len(prediction_terms)
    gold_len = len(gold_terms)
    length_ratio = min(prediction_len, gold_len) / max(prediction_len, gold_len, 1)

    return {
        "word_f1": round(word_f1, 4),
        "length_ratio": round(length_ratio, 4),
        "greeting_match": float(has_greeting(prediction) == has_greeting(gold)),
        "signoff_match": float(has_signoff(prediction) == has_signoff(gold)),
    }


def score_against_profile(prediction: str, profile: Sequence[dict]) -> dict:
    source_texts = [item.get("body", "") for item in profile if item.get("body")]
    return {
        "profile_copy_5gram_rate": max_source_ngram_overlap(prediction, source_texts, n=5),
        "profile_longest_copy_run": max_longest_common_run(prediction, source_texts),
        "style_distance": style_distance_to_profile(prediction, profile),
    }


def has_source_copy_risk(prediction: str, profile: Sequence[dict]) -> bool:
    scores = score_against_profile(prediction, profile)
    return scores["profile_copy_5gram_rate"] > 0.08 or scores["profile_longest_copy_run"] > 10


def max_source_ngram_overlap(prediction: str, sources: Sequence[str], n: int = 5) -> float:
    prediction_ngrams = set(token_ngrams(tokenize(prediction), n))
    if not prediction_ngrams:
        return 0.0

    source_ngrams = set()
    for source in sources:
        source_ngrams.update(token_ngrams(tokenize(source), n))
    if not source_ngrams:
        return 0.0
    return round(len(prediction_ngrams & source_ngrams) / len(prediction_ngrams), 4)


def token_ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def max_longest_common_run(prediction: str, sources: Sequence[str]) -> int:
    prediction_tokens = tokenize(prediction)
    if not prediction_tokens:
        return 0
    return max((longest_common_token_run(prediction_tokens, tokenize(source)) for source in sources), default=0)


def longest_common_token_run(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> int:
    if not left_tokens or not right_tokens:
        return 0
    previous = [0] * (len(right_tokens) + 1)
    best = 0
    for left_token in left_tokens:
        current = [0] * (len(right_tokens) + 1)
        for index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def style_distance_to_profile(text: str, profile: Sequence[dict]) -> float:
    profile_features = [style_features(item.get("body", "")) for item in profile if item.get("body")]
    if not profile_features:
        return 1.0
    target = style_features(text)
    averages = {
        key: sum(features[key] for features in profile_features) / len(profile_features)
        for key in target
    }
    distances = [
        normalized_distance(target["words"], averages["words"], scale=80),
        normalized_distance(target["avg_sentence_words"], averages["avg_sentence_words"], scale=25),
        normalized_distance(target["paragraphs"], averages["paragraphs"], scale=4),
        normalized_distance(target["exclamation_rate"], averages["exclamation_rate"], scale=0.4),
        normalized_distance(target["question_rate"], averages["question_rate"], scale=0.4),
        normalized_distance(target["comma_rate"], averages["comma_rate"], scale=0.15),
    ]
    return round(sum(distances) / len(distances), 4)


def style_features(text: str) -> dict:
    tokens = tokenize(text)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    sentence_words = [len(tokenize(sentence)) for sentence in sentences] or [len(tokens)]
    visible = max(len(text), 1)
    return {
        "words": len(tokens),
        "avg_sentence_words": sum(sentence_words) / max(len(sentence_words), 1),
        "paragraphs": text.count("\n\n") + 1 if text.strip() else 0,
        "exclamation_rate": text.count("!") / visible,
        "question_rate": text.count("?") / visible,
        "comma_rate": text.count(",") / visible,
    }


def normalized_distance(value: float, reference: float, scale: float) -> float:
    return min(abs(value - reference) / max(scale, 1e-9), 1.0)


def retrieve_profile_examples(
    profile: Sequence[dict], prompt: str, top_k: int = 5, mode: str = "enhanced"
) -> List[dict]:
    if (mode or "").lower() in {"basic", "keyword"}:
        return retrieve_profile_examples_basic(profile, prompt, top_k=top_k)

    prompt_terms = tokenize(prompt)
    prompt_counts = Counter(prompt_terms)
    prompt_intents = infer_prompt_intents(prompt)
    document_terms = [
        tokenize(f"{item.get('subject', '')} {item.get('body', '')}") for item in profile
    ]
    document_frequency = Counter()
    for terms in document_terms:
        document_frequency.update(set(terms))

    scored = []
    total_documents = max(len(profile), 1)
    for item, terms in zip(profile, document_terms):
        text = f"{item.get('subject', '')}\n{item.get('body', '')}"
        terms_count = Counter(terms)
        score = weighted_cosine(prompt_counts, terms_count, document_frequency, total_documents)
        overlap = len(set(prompt_terms) & set(terms))
        intent_bonus = len(prompt_intents & infer_prompt_intents(text)) * 0.18
        subject_bonus = weighted_cosine(
            prompt_counts,
            Counter(tokenize(item.get("subject", ""))),
            document_frequency,
            total_documents,
        ) * 0.4
        score = score + intent_bonus + subject_bonus + min(overlap, 8) * 0.01
        scored.append((score, item.get("id", ""), item))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item for _score, _id, item in scored[:top_k]]


def retrieve_profile_examples_basic(profile: Sequence[dict], prompt: str, top_k: int = 5) -> List[dict]:
    prompt_terms = set(tokenize(prompt))
    scored = []
    for item in profile:
        terms = set(tokenize(f"{item.get('subject', '')} {item.get('body', '')}"))
        score = len(prompt_terms & terms)
        scored.append((score, item.get("id", ""), item))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item for _score, _id, item in scored[:top_k]]


def weighted_cosine(
    left_counts: Counter, right_counts: Counter, document_frequency: Counter, total_documents: int
) -> float:
    shared = set(left_counts) & set(right_counts)
    if not shared:
        return 0.0

    def weight(term: str) -> float:
        return math.log((total_documents + 1) / (document_frequency.get(term, 0) + 1)) + 1.0

    dot = sum(left_counts[term] * right_counts[term] * weight(term) ** 2 for term in shared)
    left_norm = math.sqrt(
        sum((count * weight(term)) ** 2 for term, count in left_counts.items())
    )
    right_norm = math.sqrt(
        sum((count * weight(term)) ** 2 for term, count in right_counts.items())
    )
    return dot / max(left_norm * right_norm, 1e-9)


def infer_prompt_intents(text: str) -> set:
    lower = text.lower()
    intents = set()
    patterns = {
        "schedule": r"\b(schedule|reschedule|meeting|call|available|availability|monday|tuesday|wednesday|thursday|friday)\b",
        "review": r"\b(review|draft|agreement|comments|feedback|attached)\b",
        "deadline": r"\b(deadline|due|by\s+\w+day|need.*files|send.*files)\b",
        "thanks": r"\b(thanks|thank you|appreciate)\b",
        "question": r"\?",
        "intro": r"\b(my name|your name|who are you|introduce)\b",
        "confirm": r"\b(confirm|confirmed|approval|approve|ok|okay)\b",
        "info": r"\b(information|details|data|find|forward|send|provide)\b",
    }
    for intent, pattern in patterns.items():
        if re.search(pattern, lower):
            intents.add(intent)
    return intents


def build_style_fingerprint(profile: Sequence[dict]) -> dict:
    bodies = [item.get("body", "") for item in profile if item.get("body")]
    word_counts = [len(tokenize(body)) for body in bodies]
    first_lines = []
    last_lines = []
    short_replies = []
    identity = infer_profile_identity(profile)
    allowed_names = {part for part in identity.split() if part}
    if identity:
        allowed_names.add(identity)

    for body in bodies:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            continue
        first_lines.append(lines[0])
        closing = extract_short_signoff(lines)
        if closing and signoff_matches_identity(closing, allowed_names):
            last_lines.append(closing)
        if len(tokenize(body)) <= 20 and looks_like_authored_short_reply(body, allowed_names):
            short_replies.append(body)

    return {
        "average_words": round(sum(word_counts) / max(len(word_counts), 1), 1),
        "median_words": median_int(word_counts),
        "common_openings": top_clean_lines(first_lines, limit=4),
        "common_signoffs": top_clean_lines(last_lines, limit=4),
        "short_reply_examples": short_replies[:5],
    }


def median_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return int((sorted_values[middle - 1] + sorted_values[middle]) / 2)


def top_clean_lines(lines: Sequence[str], limit: int) -> List[str]:
    clean = []
    for line in lines:
        if 0 < len(line) <= 80 and "-----" not in line:
            clean.append(line)
    return [line for line, _count in Counter(clean).most_common(limit)]


def signoff_matches_identity(closing: str, allowed_names: set) -> bool:
    if not allowed_names:
        return False
    closing_lower = closing.lower()
    return any(name.lower() in closing_lower for name in allowed_names)


def looks_like_authored_short_reply(body: str, allowed_names: set) -> bool:
    lowered = body.lower()
    if any(token in lowered for token in ["welcome!", "refcode", "infospace", "original message"]):
        return False
    if digit_ratio(body) > 0.2:
        return False
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return False
    closing = extract_short_signoff(lines)
    return not closing or signoff_matches_identity(closing, allowed_names)


def digit_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum(char.isdigit() for char in visible) / len(visible)


def has_greeting(text: str) -> bool:
    return bool(extract_short_greeting(text))


def has_signoff(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and extract_short_signoff(lines))


def heuristic_response(
    prompt: str,
    examples: Sequence[dict],
    style_hint: str,
    identity: str = "",
) -> str:
    if is_author_style_prompt(prompt) or should_use_author_style(prompt, identity):
        return author_style_response(prompt, examples, identity=identity)

    bodies = [item.get("body", "") for item in examples if item.get("body")]
    recipient = "" if is_name_question_prompt(prompt) else extract_prompt_recipient(prompt)
    greeting = format_prompt_greeting(recipient)
    signoffs = []
    for body in bodies:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if lines:
            closing = extract_short_signoff(lines)
            if closing:
                signoffs.append(closing)
    common_signoff = Counter(signoffs).most_common(1)[0][0] if signoffs else ""

    lower_prompt = prompt.lower()
    add_signoff = True
    if is_name_question_prompt(prompt):
        incoming_name = extract_incoming_name(prompt)
        if incoming_name and identity:
            core = f"Hi {incoming_name}, my name is {identity}."
        elif identity:
            core = f"My name is {identity}."
        else:
            core = "I'm not sure what name you want me to use."
        add_signoff = False
    elif re.search(r"\bhow are you\b|\bhow are you doing\b", lower_prompt):
        core = "Doing well, thanks. Hope you are doing well too."
        add_signoff = False
    elif re.search(r"\bfiles?\b.*\b(?:wednesday|monday|tuesday|thursday|friday|deadline|due)\b|\bneed\b.*\bfiles?\b", lower_prompt):
        deadline = extract_deadline_word(prompt) or "as soon as possible"
        core = f"We need to get those files in by {deadline}. Please send them over when you can."
    elif "reschedule" in lower_prompt or "schedule" in lower_prompt:
        core = "That works for me. Please send over the updated timing and I will make it work."
    elif "confirm" in lower_prompt:
        core = "Confirmed. I will take care of it and follow up if anything changes."
    elif "review" in lower_prompt:
        core = "I will review this and get back to you with any comments."
    else:
        core = "Thanks for the note. I will take a look and get back to you shortly."

    if "brief" in style_hint or "compact" in style_hint:
        core = core.split(".")[0] + "."

    parts = []
    if greeting:
        parts.append(greeting)
    parts.append(core)
    if add_signoff and common_signoff and common_signoff.lower() not in parts[-1].lower():
        parts.append(common_signoff)
    return clean_model_response("\n\n".join(parts).strip(), identity=identity, prompt=prompt)


def author_style_response(prompt: str, examples: Sequence[dict], identity: str = "") -> str:
    topic = extract_author_topic(prompt)
    lower_topic = topic.lower()
    theme = "general"
    if re.search(r"\beducation|school|learn|learning|study|read\b", lower_topic):
        theme = "education"
        claim = "the education of the people is not a matter of ornament, but of public safety"
        contrast = "ignorance leaves a man dependent upon the loudest voice near him; knowledge gives him some power to test that voice by reason and by history"
        consequence = "by it the humblest citizen may read the story of his country, understand the value of free institutions, and judge for himself what justice requires"
        duty = "let us therefore sustain every honest means by which the mind of the child and the laboring man may be opened"
    elif re.search(r"\bfair|justice|equal|right|public life|law\b", lower_topic):
        theme = "justice"
        claim = "equal justice is the plain bond by which a free people hold together"
        contrast = "if one man may claim all benefit of the law and deny its spirit to another, he has kept the name of liberty and cast away the substance"
        consequence = "when we ask one class of men to bear what we would not bear ourselves, we loosen that bond and call it policy"
        duty = "let us measure our public conduct by the rule we would be willing to have applied to ourselves"
    elif re.search(r"\bpatient|patience|difficult|hard|week|trial\b", lower_topic):
        theme = "patience"
        claim = "patience in a season of difficulty is not surrender, but disciplined courage"
        contrast = "haste may give the appearance of strength for an hour; steadiness is the better strength for a people who mean to endure"
        consequence = "it gives honest labor time to finish its work, and keeps momentary vexation from becoming lasting injury"
        duty = "let each neighbor bear his portion calmly, neither magnifying his grievance nor forgetting another's burden"
    elif re.search(r"\breport|friday|team|finish|work|deadline\b", lower_topic):
        theme = "work"
        claim = "the work before us is plain, and the time for doing it is short"
        contrast = "no single hand can do the whole of it, and yet the failure of one hand may embarrass the labor of all the rest"
        consequence = "if each person does his part faithfully before Friday, the whole undertaking will stand in better order"
        duty = "let no man wait to be urged twice where the duty is already known"
    else:
        claim = f"the question of {topic} is one that ought to be met with candor and firmness"
        contrast = "it is easy to mistake noise for conviction, and harder to hold fast to a just purpose without anger"
        consequence = "a people cannot safely neglect small duties and yet expect large institutions to remain sound"
        duty = "let us take up the matter with charity for honest error and firmness against deliberate wrong"

    passage = build_lincoln_passage(topic, claim, contrast, consequence, duty, theme)
    return trim_text(remove_source_overlap(passage, examples), 1600)


def build_lincoln_passage(
    topic: str, claim: str, contrast: str, consequence: str, duty: str, theme: str
) -> str:
    opening = (
        f"Upon the subject of {topic}, I can only say that {claim}. "
        "The matter is not to be disposed of by a cheer, nor by a complaint, but by looking steadily at what is right and then doing it."
    )
    middle = (
        f"{contrast[0].upper() + contrast[1:]}. "
        f"{format_consequence(consequence)} "
        "This is no new doctrine, but the old and useful one, that a free people must govern themselves first in the small things if they would be trusted with the great ones."
    )
    close = (
        f"{duty[0].upper() + duty[1:]}. "
        "Let us proceed, then, without boasting and without resentment, each doing the duty nearest his hand. "
        "If we are firm in purpose, moderate in temper, and just in our dealings with one another, "
        "we shall have done more than complete the present task; we shall have strengthened the habit by which free men govern themselves."
    )
    if theme in {"justice", "education"}:
        return f"{opening}\n\n{middle}\n\n{close}"
    return f"{opening} {middle}\n\n{close}"


def extract_author_topic(prompt: str) -> str:
    lower = prompt.lower()
    if "finish their reports by friday" in lower or "finish reports by friday" in lower:
        return "finishing the reports by Friday"
    if "remain patient" in lower or "difficult week" in lower:
        return "patience during a difficult week"
    if "fairness" in lower and "public life" in lower:
        return "fairness in public life"
    if "education matters" in lower:
        return "why education matters"
    match = re.search(r"inspired by this topic:\s*(.*?)(?:\.\s*Do not quote|$)", prompt, flags=re.I | re.S)
    if match:
        return clean_author_topic(match.group(1))
    match = re.search(r"\bencouraging\s+(.+?)(?:\.|$)", prompt, flags=re.I | re.S)
    if match:
        return clean_author_topic(match.group(1))
    match = re.search(r"\b(?:about|why|that)\s+(.+?)(?:\.|$)", prompt, flags=re.I | re.S)
    if match:
        return clean_author_topic(match.group(1))
    return "this matter"


def clean_author_topic(topic: str) -> str:
    topic = re.sub(r"\s+", " ", topic).strip(" .:;\"'")
    topic = re.sub(r"^(the|a|an)\s+", "", topic, flags=re.I)
    if not topic:
        return "this matter"
    if len(topic.split()) > 14:
        topic = " ".join(topic.split()[:14])
    return topic


def format_consequence(consequence: str) -> str:
    consequence = consequence.strip()
    if not consequence:
        return ""
    if consequence.lower().startswith(("for ", "when ", "if ", "by ")):
        return consequence[0].upper() + consequence[1:] + "."
    return f"For {consequence}."


def remove_source_overlap(text: str, examples: Sequence[dict]) -> str:
    cleaned = text
    source_phrases = set()
    for item in examples:
        tokens = tokenize(item.get("body", ""))
        source_phrases.update(" ".join(ngram) for ngram in token_ngrams(tokens, 6))
    for phrase in source_phrases:
        if phrase and phrase in cleaned.lower():
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.I)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def extract_deadline_word(prompt: str) -> str:
    match = re.search(
        r"\b(?:by|before|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        prompt,
        flags=re.I,
    )
    if match:
        return match.group(1).capitalize()
    return ""


def looks_like_failed_generation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(tokenize(stripped)) < 3 and len(stripped) < 32:
        return True
    visible = [char for char in stripped if not char.isspace()]
    if not visible:
        return True
    ascii_letters = sum(char.isascii() and char.isalpha() for char in visible)
    cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in visible)
    if cjk_chars and cjk_chars / len(visible) > 0.2:
        return True
    if ascii_letters / len(visible) < 0.35 and len(visible) > 8:
        return True
    words = tokenize(stripped)
    if len(words) >= 12:
        most_common_count = Counter(words).most_common(1)[0][1]
        if most_common_count / len(words) > 0.35:
            return True
    return False


def infer_profile_identity(profile: Sequence[dict]) -> str:
    excluded = {
        "thanks",
        "regards",
        "best",
        "hello",
        "original message",
        "forwarded by",
        "subject",
        "from",
        "to",
    }
    candidates = []
    for item in profile:
        lines = [line.strip() for line in item.get("body", "").splitlines() if line.strip()]
        for line in lines[-5:]:
            normalized = line.strip(" .,!:-")
            lower_normalized = normalized.lower()
            if lower_normalized in excluded or "original message" in lower_normalized:
                continue
            if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", normalized):
                candidates.append(normalized)
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def clean_model_response(text: str, identity: str = "", prompt: str = "") -> str:
    text = text.strip()
    text = re.sub(r"(?i)^response:\s*", "", text).strip()
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)
    text = re.sub(r",\s*([!?])", r"\1", text)
    paragraphs = re.split(r"\n\s*\n", text, maxsplit=1)
    if len(paragraphs) == 2 and re.search(
        r"(?i)\b(here it is|supposed to write|email body|reply as|draft)\b", paragraphs[0]
    ):
        text = paragraphs[1].strip()
    lines = []
    for line in text.splitlines():
        if re.match(r"(?i)^\s*subject\s*:", line):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines).strip()
    text = remove_wrong_signature(text, identity)
    text = remove_bad_topic_greeting(text, prompt)
    return text


def remove_wrong_signature(text: str, identity: str) -> str:
    if not identity:
        return text
    allowed = {part.lower() for part in identity.split()} | {identity.lower()}
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return text
    last = lines[-1].strip().strip(".,")
    if re.fullmatch(r"[A-Z][A-Za-z'-]{1,30}", last) and last.lower() not in allowed:
        lines.pop()
        if lines and re.fullmatch(r"(?i)(thanks|regards|best|sincerely|cheers)[,.!]?", lines[-1].strip()):
            lines.pop()
    return "\n".join(lines).strip()


def remove_bad_topic_greeting(text: str, prompt: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    if not re.fullmatch(r"[A-Z][A-Za-z'-]{2,30},", first):
        return text
    name = first[:-1].lower()
    non_person_terms = {"japan", "eol", "ebs", "bandwidth", "legal", "online"}
    if name in non_person_terms:
        return "\n".join(lines[1:]).strip()
    return text


def extract_prompt_recipient(prompt: str) -> str:
    patterns = [
        r"\b(?:to|for)\s+([A-Z][a-z]+)\b",
        r"\b(?:from|email from|reply to)\s+([A-Z][a-z]+)\b",
        r"^([A-Z][a-z]+),",
    ]
    stopwords = {"This", "That", "Email", "Message", "Reply", "User", "Style"}
    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match and match.group(1) not in stopwords:
            return match.group(1)
    return ""


def extract_incoming_name(prompt: str) -> str:
    patterns = [
        r"(?i)\bmy name(?: is|'s|s)\s+([a-z]+)\b",
        r"(?i)\bI'm\s+([a-z]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).title()
    return ""


def format_prompt_greeting(recipient: str) -> str:
    if not recipient:
        return ""
    return f"{recipient},"


def extract_short_greeting(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line or len(first_line) > 40:
        return ""

    greeting_patterns = [
        r"(?i)(hi|hello|hey|dear)\s+[A-Za-z][A-Za-z .'-]{0,30},?",
        r"(?i)(hi|hello|hey),?",
        r"[A-Z][A-Za-z .'-]{0,30},",
    ]
    if any(re.fullmatch(pattern, first_line) for pattern in greeting_patterns):
        return first_line
    return ""


def extract_short_signoff(lines: Sequence[str]) -> str:
    if len(lines) >= 2 and re.fullmatch(r"(?i)(thanks|thank you|regards|best|sincerely|cheers)[,.!]?", lines[-2]):
        if len(lines[-1]) <= 40 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,39}", lines[-1]):
            return f"{lines[-2]}\n{lines[-1]}"

    last = lines[-1]
    if len(last) <= 50 and re.fullmatch(r"(?i)(thanks|thank you|regards|best|sincerely|cheers)[,.!]?", last):
        return last
    return ""


def email_to_schema(record: EmailRecord) -> dict:
    return {"id": record.id, "subject": record.subject, "body": record.body}


def build_query_prompt(record: EmailRecord) -> str:
    subject = record.subject or "(no subject)"
    if record.context:
        return (
            "Write a reply to this email in the user's style.\n\n"
            f"Subject: {subject}\n\n"
            f"Incoming email:\n{record.context}"
        )
    return f"Write an email in the user's style for this subject: {subject}"


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _safe_get_content(part) -> str:
    try:
        content = part.get_content()
        return content if isinstance(content, str) else str(content)
    except Exception:
        payload = part.get_payload(decode=True) or b""
        return payload.decode(part.get_content_charset() or "utf-8", errors="ignore")


def _looks_like_sent_path(path: Path, maildir: Path) -> bool:
    try:
        parts = path.relative_to(maildir).parts
    except ValueError:
        return False
    return _looks_like_sent_parts(parts)


def _looks_like_sent_parts(parts: Sequence[str]) -> bool:
    parts = [part.lower() for part in parts]
    return any(part in SENT_DIR_NAMES for part in parts)


def _find_emails_csv(path: Path) -> Optional[Path]:
    if path.is_file() and path.name.lower().endswith(".csv"):
        return path
    candidate = path / "emails.csv"
    if candidate.exists():
        return candidate
    return None
