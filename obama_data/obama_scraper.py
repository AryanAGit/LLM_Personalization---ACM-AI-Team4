"""
scrape_obama_speeches.py
------------------------
Scrapes the Barack Obama speech archive from americanrhetoric.com and saves
all transcripts to obama_speeches.json.

Requirements:
    pip install requests beautifulsoup4 lxml

Usage:
    python scrape_obama_speeches.py

Output:
    obama_speeches.json  — array of speech objects:
        {
          "title":     "...",
          "url":       "https://...",
          "date":      "...",   # extracted from page when available
          "transcript": "..."
        }
    scrape_errors.json   — any URLs that failed, for retry
"""

import json
import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_URL   = "https://www.americanrhetoric.com/barackobamaspeeches.htm"
BASE_URL    = "https://www.americanrhetoric.com/"
OUT_FILE    = "obama_speeches.json"
ERR_FILE    = "scrape_errors.json"

DELAY       = 1.2      # seconds between requests (be polite)
TIMEOUT     = 20       # seconds per request
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.americanrhetoric.com/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get(url: str, session: requests.Session) -> requests.Response | None:
    """GET with retries; returns None on persistent failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, e)
            if attempt < MAX_RETRIES:
                time.sleep(DELAY * attempt * 2)
    return None


def is_speech_url(href: str) -> bool:
    """
    Filter index links to only those that point to actual speech pages.
    americanrhetoric.com speech pages follow patterns like:
        /speeches/barackobama*.htm
        /speeches/barackobama*.html
    but NOT external links, audio files, PDFs, or anchor-only hrefs.
    """
    if not href:
        return False
    href_lower = href.lower()
    # skip fragments, mailto, javascript
    if href.startswith(("#", "mailto:", "javascript:")):
        return False
    # skip media files
    if any(href_lower.endswith(ext) for ext in (".mp3", ".mp4", ".pdf", ".jpg", ".png", ".gif")):
        return False
    # must contain "obama" (case-insensitive) to be a speech page
    if "obama" not in href_lower:
        return False
    # must be an htm/html page
    if not (href_lower.endswith(".htm") or href_lower.endswith(".html")):
        return False
    return True


def extract_speech_links(soup: BeautifulSoup) -> list[dict]:
    """Return list of {title, url} dicts from the index page."""
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not is_speech_url(href):
            continue
        full_url = urljoin(BASE_URL, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        title = a.get_text(separator=" ", strip=True) or href
        links.append({"title": title, "url": full_url})

    return links


def extract_date(soup: BeautifulSoup, url: str) -> str:
    """
    Try several heuristics to pull the speech date from the page.
    Falls back to an empty string.
    """
    # 1. Look for a <p> or <div> containing a date-like pattern near the top
    date_re = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    )
    for tag in soup.find_all(["p", "div", "span", "td"], limit=60):
        text = tag.get_text()
        m = date_re.search(text)
        if m:
            return m.group(0)

    # 2. Try to pull a year from the URL itself
    m = re.search(r"(200[0-9]|201[0-9]|202[0-9])", url)
    if m:
        return m.group(1)

    return ""


def extract_transcript(soup: BeautifulSoup) -> str:
    """
    Extract the main speech transcript text.
    americanrhetoric pages vary in structure; we try several strategies.
    """
    # Strategy 1: look for a div with id or class containing "transcript" / "speech"
    for attr_val in ["transcript", "speech", "maincontent", "main-content", "content"]:
        for tag in soup.find_all(["div", "article", "section"], id=re.compile(attr_val, re.I)):
            text = tag.get_text(separator="\n", strip=True)
            if len(text) > 300:
                return clean_text(text)
        for tag in soup.find_all(["div", "article", "section"], class_=re.compile(attr_val, re.I)):
            text = tag.get_text(separator="\n", strip=True)
            if len(text) > 300:
                return clean_text(text)

    # Strategy 2: find the largest contiguous block of <p> tags
    paragraphs = soup.find_all("p")
    if paragraphs:
        text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        if len(text) > 300:
            return clean_text(text)

    # Strategy 3: strip nav/header/footer and take body text
    for unwanted in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        unwanted.decompose()
    body = soup.find("body")
    if body:
        return clean_text(body.get_text(separator="\n", strip=True))

    return clean_text(soup.get_text(separator="\n", strip=True))


def clean_text(text: str) -> str:
    """Normalise whitespace; remove runs of blank lines."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    # ── 1. Fetch index ────────────────────────────────────────────────────────
    log.info("Fetching index: %s", INDEX_URL)
    r = get(INDEX_URL, session)
    if r is None:
        log.error("Could not fetch index page. Aborting.")
        return

    soup = BeautifulSoup(r.text, "lxml")
    speech_links = extract_speech_links(soup)
    log.info("Found %d unique speech links on index page.", len(speech_links))

    if not speech_links:
        log.error("No speech links found — the page structure may have changed.")
        return

    # ── 2. Scrape each speech ─────────────────────────────────────────────────
    speeches = []
    errors   = []

    for i, link in enumerate(speech_links, 1):
        title = link["title"]
        url   = link["url"]
        log.info("[%d/%d] Scraping: %s", i, len(speech_links), url)

        time.sleep(DELAY)
        r = get(url, session)

        if r is None:
            log.warning("  ✗ Failed — adding to error list.")
            errors.append({"title": title, "url": url})
            continue

        page_soup = BeautifulSoup(r.text, "lxml")
        date      = extract_date(page_soup, url)
        transcript = extract_transcript(page_soup)

        if len(transcript) < 100:
            log.warning("  ⚠ Very short transcript (%d chars) — may be a stub.", len(transcript))

        speeches.append({
            "title":      title,
            "url":        url,
            "date":       date,
            "transcript": transcript,
        })

        log.info("  ✓ %d chars | date: %s", len(transcript), date or "(unknown)")

        # Checkpoint every 50 speeches so you don't lose progress
        if i % 50 == 0:
            _save(speeches, OUT_FILE)
            log.info("  💾 Checkpoint saved (%d speeches so far).", len(speeches))

    # ── 3. Save final output ──────────────────────────────────────────────────
    _save(speeches, OUT_FILE)
    log.info("Done. %d speeches saved to %s.", len(speeches), OUT_FILE)

    if errors:
        _save(errors, ERR_FILE)
        log.warning("%d URLs failed — see %s for retry.", len(errors), ERR_FILE)


def _save(data, path: str):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()