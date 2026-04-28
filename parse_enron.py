import pandas as pd
import json
from email import policy
from email.parser import Parser

CSV_PATH = "enron_data/emails.csv"
MIN_EMAILS_PER_USER = 100


# -----------------------------
# 1. Parse raw email string
# -----------------------------
def parse_raw_email(raw_text):
    try:
        msg = Parser(policy=policy.default).parsestr(raw_text)
    except:
        return None

    return {
        "message_id": msg.get("Message-ID", ""),
        "date": msg.get("Date", ""),
        "from": msg.get("From", "").strip().lower(),
        "subject": msg.get("Subject", "").strip(),
        "body": extract_body(msg)
    }


def extract_body(msg):
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    parts.append(part.get_content())
                except:
                    pass
        return "\n".join(parts).strip()
    else:
        try:
            return msg.get_content().strip()
        except:
            return ""


# -----------------------------
# 2. Load + parse dataset
# -----------------------------
def load_dataset(path):
    df = pd.read_csv(path, nrows=30000)

    parsed_rows = []

    for _, row in df.iterrows():
        parsed = parse_raw_email(row["message"])
        if parsed and parsed["from"] and parsed["body"]:
            parsed["file"] = row["file"]
            parsed_rows.append(parsed)

    return pd.DataFrame(parsed_rows)


# -----------------------------
# 3. Group users
# -----------------------------
def group_users(df):
    grouped = df.groupby("from")

    users = {
        user: group.sort_values("date")
        for user, group in grouped
        if len(group) >= MIN_EMAILS_PER_USER
    }

    return users


# -----------------------------
# 4. Build queries (reply detection)
# -----------------------------
def normalize_subject(subj):
    if not subj:
        return ""
    subj = subj.lower().strip()

    # remove common prefixes
    for prefix in ["re:", "fw:", "fwd:"]:
        if subj.startswith(prefix):
            return subj[len(prefix):].strip()

    return subj


def build_queries(user_df):
    queries = []

    subject_map = {}

    for _, row in user_df.iterrows():
        base = normalize_subject(row["subject"])
        subject_map.setdefault(base, []).append(row)

    for _, row in user_df.iterrows():
        subj = row["subject"].lower()

        if subj.startswith("re:"):
            base = normalize_subject(subj)

            if base in subject_map:
                original = subject_map[base][0]

                queries.append({
                    "id": row["message_id"] or row["file"],
                    "input": f"Write a reply to this email in the user's style: '{original['body'][:2000]}'",
                    "gold": row["body"]
                })

    return queries


# -----------------------------
# 5. Style generation (LLM hook)
# -----------------------------
def generate_style_description(bodies):
    sample = "\n\n".join(b[:1000] for b in bodies[:20])

    prompt = f"""
Analyze the following emails and describe the writer's tone and style:

{sample}
"""

    # TODO: replace with actual LLM call
    return "Placeholder: direct, concise, business-oriented tone."


# -----------------------------
# 6. Build dataset
# -----------------------------
def build_dataset(users):
    dataset = []
    styles = []

    for idx, (user, df) in enumerate(users.items()):
        df = df.reset_index(drop=True)

        profile = [
            {
                "id": row["message_id"] or row["file"],
                "subject": row["subject"],
                "body": row["body"]
            }
            for _, row in df.iterrows()
        ]

        queries = build_queries(df)

        dataset.append({
            "user_id": idx,
            "profile": profile,
            "query": queries
        })

        styles.append({
            "id": idx,
            "output": generate_style_description(df["body"].tolist())
        })

    return dataset, styles


# -----------------------------
# 7. Save
# -----------------------------
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    df = load_dataset(CSV_PATH)
    users = group_users(df)

    dataset, styles = build_dataset(users)

    save_json(dataset, "enron_users.json")
    save_json(styles, "enron_styles.json")