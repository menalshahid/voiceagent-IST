#!/usr/bin/env python3
"""
Update all_kb.txt with:
  1) Full faculty directory (ist_scraped_data/faculty_directory.txt)
  2) Latest announcements from ist.edu.pk + scraped news flat files

Run after scraping or before deploy:
  python kb_update.py
  python kb_update.py --no-fetch   # faculty only, skip live web fetch
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "all_kb.txt"
FACULTY_PATH = ROOT / "ist_scraped_data" / "faculty_directory.txt"
FLAT_NEWS = ROOT / "ist_scraped_data" / "flat_text"
BACKUP_DIR = ROOT / "backup_kb"

FACULTY_OLD_START = "## FACULTY AND PROGRAMS BY DEPARTMENT"
FACULTY_OLD_END = "## PROGRAMS AND ADMISSIONS DATA"
FACULTY_MARKER_START = "## [IST_FACULTY_DIRECTORY_START]"
FACULTY_MARKER_END = "## [IST_FACULTY_DIRECTORY_END]"

ANNOUNCE_START = "## [IST_ANNOUNCEMENTS_START]"
ANNOUNCE_END = "## [IST_ANNOUNCEMENTS_END]"

ANNOUNCE_URLS = (
    "https://ist.edu.pk/",
    "https://ist.edu.pk/news-events",
)


def fetch_page_text(url: str) -> str | None:
    try:
        import trafilatura
    except ImportError:
        print("pip install trafilatura", file=sys.stderr)
        return None
    try:
        html = trafilatura.fetch_url(url)
        if not html:
            return None
        return (trafilatura.extract(html, include_tables=True, favor_recall=True) or "").strip()
    except Exception as exc:
        print(f"  fetch error {url}: {exc}")
        return None


def load_faculty_block() -> str:
    if not FACULTY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {FACULTY_PATH} — run: python build_faculty_directory.py"
        )
    text = FACULTY_PATH.read_text(encoding="utf-8").strip()
    if FACULTY_MARKER_START not in text:
        text = f"{FACULTY_MARKER_START}\n{text}\n{FACULTY_MARKER_END}"
    return text + "\n"


def build_announcements_block(fetch_live: bool = True) -> str:
    parts: list[str] = [
        ANNOUNCE_START,
        "# IST ANNOUNCEMENTS & NEWS (ist.edu.pk)",
        f"# Updated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "Use this section for latest admissions, events, workshops, and university news.",
        "",
    ]

    if fetch_live:
        for url in ANNOUNCE_URLS:
            time.sleep(0.8)
            body = fetch_page_text(url)
            if body and len(body) > 60:
                parts.append("=" * 70)
                parts.append(f"[TOPIC: Announcements — {url}]")
                parts.append(body)
                parts.append("")

    if FLAT_NEWS.is_dir():
        news_files = sorted(FLAT_NEWS.glob("news_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        parts.append("=" * 70)
        parts.append("[TOPIC: Scraped news article pages]")
        for fp in news_files[:25]:
            raw = fp.read_text(encoding="utf-8", errors="replace")
            m_url = re.search(r"^URL:\s*(.+)$", raw, re.M)
            m_title = re.search(r"^TITLE:\s*(.+)$", raw, re.M)
            url = m_url.group(1).strip() if m_url else fp.stem
            title = m_title.group(1).strip() if m_title else ""
            if "--- CONTENT ---" in raw:
                body = raw.split("--- CONTENT ---", 1)[1].strip()
            else:
                body = raw
            if len(body) < 40:
                continue
            parts.append(f"\n--- {title or url} ---")
            parts.append(f"URL: {url}")
            parts.append(body[:4000])
        parts.append("")

    parts.append(ANNOUNCE_END)
    return "\n".join(parts).strip() + "\n"


def replace_faculty_section(kb: str, faculty_block: str) -> str:
    """Replace old abbreviated faculty section with full directory."""
    if FACULTY_MARKER_START in kb and FACULTY_MARKER_END in kb:
        i0 = kb.index(FACULTY_MARKER_START)
        i1 = kb.index(FACULTY_MARKER_END) + len(FACULTY_MARKER_END)
        kb = kb[:i0] + faculty_block.strip() + "\n\n" + kb[i1:].lstrip()

    if FACULTY_OLD_START in kb and FACULTY_OLD_END in kb:
        i0 = kb.index(FACULTY_OLD_START)
        i1 = kb.index(FACULTY_OLD_END)
        kb = kb[:i0] + faculty_block.strip() + "\n\n" + kb[i1:]
    elif FACULTY_MARKER_START not in kb:
        anchor = FACULTY_OLD_END
        if anchor in kb:
            kb = kb.replace(anchor, faculty_block.strip() + "\n\n" + anchor, 1)
        else:
            kb = faculty_block + "\n\n" + kb
    return kb


def replace_announcements_section(kb: str, announce_block: str) -> str:
    if ANNOUNCE_START in kb and ANNOUNCE_END in kb:
        i0 = kb.index(ANNOUNCE_START)
        i1 = kb.index(ANNOUNCE_END) + len(ANNOUNCE_END)
        return kb[:i0] + announce_block.strip() + "\n\n" + kb[i1:].lstrip()

    insert_before = FACULTY_OLD_END
    if insert_before not in kb:
        insert_before = "## [AUTO_SCRAPED_IST_WEB_START]"
    if insert_before in kb:
        i = kb.index(insert_before)
        return kb[:i] + announce_block.strip() + "\n\n" + kb[i:]
    return kb + "\n\n" + announce_block


def update_kb(kb_path: Path = KB_PATH, fetch_live: bool = True) -> None:
    if not kb_path.is_file():
        raise FileNotFoundError(kb_path)

    kb = kb_path.read_text(encoding="utf-8")
    faculty = load_faculty_block()
    kb = replace_faculty_section(kb, faculty)
    announce = build_announcements_block(fetch_live=fetch_live)
    kb = replace_announcements_section(kb, announce)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"all_kb_{stamp}.txt"
    backup.write_text(kb_path.read_text(encoding="utf-8"), encoding="utf-8")

    kb_path.write_text(kb, encoding="utf-8")
    print(f"Updated {kb_path} ({len(kb):,} chars)")
    print(f"  Backup: {backup}")
    print(f"  Faculty: {FACULTY_PATH.name}")
    print(f"  Announcements: live={fetch_live}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="Skip live ist.edu.pk fetch")
    ap.add_argument("--kb", type=Path, default=KB_PATH)
    args = ap.parse_args()
    try:
        update_kb(args.kb, fetch_live=not args.no_fetch)
    except Exception as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
