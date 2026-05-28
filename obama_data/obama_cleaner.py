"""
clean_transcripts.py
--------------------
Cleans the obama_speeches.json transcripts by stripping all site boilerplate
and isolating only the speech text.

Usage:
    python clean_transcripts.py
    python clean_transcripts.py --input obama_speeches.json --output obama_speeches_clean.json

Output:
    obama_speeches_clean.json   — same structure, with a cleaned `transcript` field
    clean_report.txt            — per-speech summary (chars removed, issues flagged)
"""

import json
import re
import argparse
from pathlib import Path

# ── Boilerplate patterns ──────────────────────────────────────────────────────
#
# Based on inspection, every page has:
#
#  HEADER (before speech text):
#   - "B arack O bama" (spaced-out name, sometimes varies)
#   - Page/speech title line
#   - "delivered <date>, <location>"
#   - Media/player lines: "Your browser does not support...", "Audio mp3...",
#     "Audio AR-XE mp3...", "click \nfor pdf", "Brief Audio mp3 Clip..."
#   - Authenticity block: "[AUTHENTICITY CERTIFIED: ...]" or "[text unauthenticated]"
#   - Footnote markers embedded in text (superscript numbers like \n1\n)
#
#  FOOTER (after speech text):
#   - "Book/CDs by Michael E. Eidenmuller..."
#   - "Text Source :", "Text & Audio Source :", "Audio Source :", etc.
#   - "Audio Note : AR-XE = ..."
#   - "Video Note : AI upscaled ..."
#   - "Image Source :"
#   - "Page Updated :" / "Page Update :"
#   - "U.S. Copyright Status :"
#   - "Top 100 American Speeches", "Online Speech Bank", "Movie Speeches"
#   - "© Copyright 2001-Present. American Rhetoric..."
#   - "Also in this database : ..."
#   - Footnote definitions (¹ ² ³ ...)
#
#  IN-TEXT noise:
#   - Stage directions / editor notes in [brackets] — keep these, they're
#     part of the transcript (e.g. "[applause]", "[to audience on left]")
#   - Lone superscript-style numbers on their own line (footnote refs): remove
#   - "\xa0" (non-breaking space): normalise to regular space
#   - "\x92" (Windows-1252 right-apostrophe): normalise to '
#   - Spaced name header: "B arack O bama" → remove
#

# Lines that are purely boilerplate and should be dropped
JUNK_LINE_PATTERNS = [
    # media players
    r"^Your browser does not support (the audio element|the video tag)\.?$",
    r"^(Brief\s+)?Audio(\s+AR-XE)?\s+mp3(\s+(Clip\s+)?of Address)?\.?$",
    r"^Audio\s+AR-XE\s+mp3\s+of Address\.?$",
    r"^click\s*\nfor pdf$",          # handled in multiline strip too
    r"^click$",
    r"^for pdf$",
    # authenticity block header lines
    r"^\[AUTHENTICITY CERTIFIED:.*\]$",
    r"^\[text\s+unauthenticated\]$",
    r"^\[Text not yet authenticated\]?\.?$",
    # name header variants
    r"^B\s+arack\s+O\s+bama$",
    r"^Barack\s+Obama$",
    # footer anchors / nav links
    r"^Top\s+100\s+American\s+Speeches$",
    r"^Online\s+Speech\s+Bank$",
    r"^Movie\s+Speeches$",
    r"^© Copyright \d{4}.*American Rhetoric.*$",
    r"^Book/CDs by Michael E\. Eidenmuller.*$",
    r"^Published by\s*$",
    r"^McGraw-Hill.*$",
    # source / update / copyright lines
    r"^(Text|Audio|Video|Image|Text\s+&\s+Audio)\s+(Source|Note)\s*:.*$",
    r"^Audio\s+Note\s*:.*$",
    r"^Video\s+Note\s*:.*$",
    r"^Image\s+Source\s*:.*$",
    r"^Page\s+Update[d]?\s*:.*$",
    r"^U\.S\.\s+Copyright\s+Status\s*:.*$",
    r"^Also in this database\s*:.*$",
    r"^(A\s+)?merican\s*(R\s*)?hetoric.*$",  # fragmented "American Rhetoric" from split spans
    # lone footnote reference numbers on their own line
    r"^\d+$",
    # blank/whitespace only
    r"^\s*$",
]

# Multiline / span patterns to strip before line-splitting
MULTILINE_STRIP = [
    # authenticity block (possibly multi-line, square or round brackets, optional leading [)
    r"\[?\s*AUTHENTICITY CERTIFIED:[^\]]*\]",
    r"\[?\s*AUTHENTICITY CERTIFIED:[^\n]*\n[^\]]*\]",
    r"\[text\s*\n\s*unauthenticated\]",
    r"\[Text not yet authenticated\]",
    # "click \nfor pdf" across a newline
    r"click\s*\n\s*for pdf",
    # "B arack O bama" header (handles mid-transcript occurrences too)
    r"B\s+arack\s+O\s+bama",
    # superscript footnote numbers inline (e.g. "faithfully\n1\n, and")
    # only strip when the number is surrounded by newlines (i.e. on its own line)
    r"\n\d{1,2}\n",
    # stray audio player lines that weren't caught by header/footer sentinels
    r"(English Language\s+|Japanese Language\s+)?Audio(\s+mp3)?(\s+AR-XE)?(\s+mp3)?\s+of Address",
    r"Audio\s+AR-XE\s+mp3\s+of Address",
]

# Footer sentinel — everything from here to end of string is boilerplate
# We stop at the earliest match of any of these patterns
FOOTER_SENTINELS = [
    r"Book/CDs by Michael E\. Eidenmuller",
    r"(Original\s+)?Text\s*(,\s*(Audio|Video|Image)[^:]{0,40})?\s+Source\s*:",
    r"(Original\s+)?(Audio|Video|Image)[^:]{0,60}Source\s*:",
    r"Available at Amazon",
    r"Text\s*&\s*Audio\s+Source\s*:",
    r"Audio\s+Source\s*:",
    r"Page\s+Update[d]?\s*:",
    r"Page\s+Created\s*:",
    r"U\.S\.\s+Copyright\s+Status\s*:",
    r"Copyright\s+Status\s*:",
    r"© Copyright \d{4}",
    r"Also in this database\s*:",
    r"Top\s+100\s+American\s+Speeches",
    r"Online\s+Speech\s+Bank",
]

# Header sentinel — the speech text begins AFTER the last match of these
# in the preamble region (first ~600 chars)
HEADER_SENTINELS = [
    r"\[AUTHENTICITY CERTIFIED:[^\]]*\]",
    r"\[text\s*\n?\s*unauthenticated\]",
    r"\[Text not yet authenticated\]",
    # If no authenticity block, the "delivered <date>" line is the last header line
    r"delivered\s+\d{1,2}\s+\w+\s+\d{4}",
    r"Delivered\s+\d{1,2}\s+\w+\s+\d{4}",
]

# ── Character normalisation ───────────────────────────────────────────────────

def normalise_chars(text: str) -> str:
    replacements = {
        "\xa0":  " ",   # non-breaking space
        "\x92":  "'",   # Windows-1252 right single quote
        "\x91":  "'",   # Windows-1252 left single quote
        "\x93":  '"',   # Windows-1252 left double quote
        "\x94":  '"',   # Windows-1252 right double quote
        "\x96":  "–",   # Windows-1252 en-dash
        "\x97":  "—",   # Windows-1252 em-dash
        "\r\n":  "\n",
        "\r":    "\n",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


# ── Main cleaning pipeline ────────────────────────────────────────────────────

def clean(transcript: str) -> str:
    text = normalise_chars(transcript)

    # 1. Strip multiline patterns
    for pat in MULTILINE_STRIP:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    # 2. Chop footer: find earliest sentinel, keep everything before it
    footer_match = None
    for pat in FOOTER_SENTINELS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m and (footer_match is None or m.start() < footer_match.start()):
            footer_match = m
    if footer_match:
        text = text[:footer_match.start()]

    # 3. Chop header: find the end of the last header sentinel in the first
    #    600 chars (to avoid matching actual speech content)
    header_end = 0
    preamble = text[:700]
    for pat in HEADER_SENTINELS:
        for m in re.finditer(pat, preamble, flags=re.IGNORECASE | re.DOTALL):
            if m.end() > header_end:
                header_end = m.end()

    # If no sentinel found, fall back to stripping the first block of lines
    # that look like title/date/media metadata (before first real paragraph)
    if header_end == 0:
        text = strip_leading_metadata(text)
    else:
        text = text[header_end:]

    # 3b. Strip any residual location fragment left at the very start
    #     e.g. ", Chicago, \nIllinois\n"  or ", Fleet Center, Boston\n"
    text = re.sub(r"^\s*,\s*[^\n]{0,120}\n", "", text, count=1)

    # 4. Drop junk lines
    lines = text.split("\n")
    cleaned_lines = []
    junk_re = [re.compile(p, re.IGNORECASE) for p in JUNK_LINE_PATTERNS]
    for line in lines:
        stripped = line.strip()
        if any(r.match(stripped) for r in junk_re):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 5. Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def strip_leading_metadata(text: str) -> str:
    """
    Fallback: drop leading lines that look like metadata (title, date, media
    notices) until we hit a line that looks like the start of actual speech.
    A 'real' line is one that starts with a capital letter, has more than
    6 words, and doesn't match obvious metadata patterns.
    """
    lines = text.split("\n")
    meta_patterns = [
        re.compile(r, re.IGNORECASE) for r in [
            r"^delivered\b",
            r"^Delivered\b",
            r"^\[",
            r"^Audio\b",
            r"^Video\b",
            r"^click\b",
            r"^for pdf",
            r"^B\s+arack",
            r"^Barack Obama",
            r"^\d{4}$",
            r"^\s*$",
        ]
    ]
    for i, line in enumerate(lines):
        stripped = line.strip()
        word_count = len(stripped.split())
        is_meta = any(p.match(stripped) for p in meta_patterns)
        if not is_meta and word_count >= 5 and stripped[0:1].isupper():
            return "\n".join(lines[i:])
    return text


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean Obama speech transcripts")
    parser.add_argument("--input",  default="obama_speeches.json")
    parser.add_argument("--output", default="obama_speeches_clean.json")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} speeches from {args.input}")

    results = []
    report_lines = []
    skipped = 0

    for i, speech in enumerate(data):
        raw = speech.get("transcript", "")
        cleaned = clean(raw)

        char_removed = len(raw) - len(cleaned)
        pct_removed  = (char_removed / max(len(raw), 1)) * 100

        flag = ""
        if len(cleaned) < 200:
            flag = "⚠ VERY SHORT"
            skipped += 1
        elif pct_removed > 60:
            flag = "⚠ LARGE REDUCTION"

        report_lines.append(
            f"[{i:>3}] {speech['title'][:55]:<55} "
            f"raw={len(raw):>6}  clean={len(cleaned):>6}  "
            f"-{pct_removed:4.0f}%  {flag}"
        )

        results.append({**speech, "transcript": cleaned})

    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(results)} speeches to {args.output}")

    report_path = "clean_report.txt"
    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved to {report_path}")
    print(f"Speeches flagged as very short: {skipped}")


if __name__ == "__main__":
    main()