"""Deterministic faculty answers from IST_FACULTY_DIRECTORY in all_kb.txt."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

MARKER_START = "## [IST_FACULTY_DIRECTORY_START]"
MARKER_END = "## [IST_FACULTY_DIRECTORY_END]"

# Query phrases -> canonical department header in directory
DEPT_ALIASES: list[tuple[tuple[str, ...], str]] = [
    (("computing", "computer science", " comp ", " cs ", "software engineering", "data science", " hod comp", "hod.comp"), "Department of Computer Science"),
    (("electrical", " ee ", "computer engineering dept"), "Department of Electrical Engineering"),
    (("avionics",), "Department of Avionics Engineering"),
    (("aeronautic", "aerospace", "astro",), "Department of Aeronautics & Astronautics"),
    (("mechanical", " mech"), "Department of Mechanical Engineering"),
    (("material", "metallurgy", "biotech"), "Department of Materials Science"),
    (("space science", "physics", "remote sensing", "chemistry"), "Department of Space Science"),
    (("mathematics", "applied math", "statistics", " am&s"), "Department of Applied Mathematics & Statistics"),
    (("humanities",), "Department of Humanities & Sciences"),
]

_by_dept: dict[str, list[dict[str, str]]] = {}


def _valid_name(name: str) -> bool:
    n = (name or "").strip()
    if len(n) < 4:
        return False
    if n.startswith("BS(") or n.startswith("MS("):
        return False
    if "@" in n or "http" in n.lower():
        return False
    return True


def _parse_directory_block(block: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    parts = re.split(r"^## DEPARTMENT:\s*", block, flags=re.MULTILINE)
    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        dept = lines[0].strip()
        members: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in part.splitlines():
            if line.startswith("--- Faculty #"):
                if current.get("name") and _valid_name(current["name"]):
                    members.append(current)
                current = {}
                continue
            m = re.match(r"^\s*Name:\s*(.+)$", line)
            if m:
                current["name"] = m.group(1).strip()
                continue
            m = re.match(r"^\s*Designation:\s*(.+)$", line)
            if m:
                current["designation"] = m.group(1).strip()
                continue
            m = re.match(r"^\s*Email:\s*(.+)$", line)
            if m and "@" in m.group(1):
                current["email"] = m.group(1).strip()
        if current.get("name") and _valid_name(current["name"]):
            members.append(current)
        if members:
            out[dept] = members
    return out


def reload(kb_path: str | Path = "all_kb.txt") -> None:
    global _by_dept
    path = Path(kb_path)
    if not path.is_file():
        _by_dept = {}
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    if MARKER_START not in raw or MARKER_END not in raw:
        _by_dept = {}
        return
    i0 = raw.index(MARKER_START) + len(MARKER_START)
    i1 = raw.index(MARKER_END)
    _by_dept = _parse_directory_block(raw[i0:i1])


def detect_department(question: str) -> str | None:
    q = f" {question.lower()} "
    best: str | None = None
    best_score = 0
    for phrases, dept in DEPT_ALIASES:
        score = sum(1 for p in phrases if p in q)
        if score > best_score:
            best_score = score
            best = dept
    return best if best_score > 0 else None


def is_faculty_list_query(question: str) -> bool:
    q = question.lower()
    if not any(w in q for w in ("faculty", "professor", "lecturer", "teacher", "staff", "hod", "head of department")):
        return False
    return any(w in q for w in ("list", "name", "who", "all", "members", "department", "rank", "designation"))


def format_faculty_answer(dept: str, members: list[dict[str, str]], language: str) -> str:
    """Voice-friendly grouped by designation; KB-accurate names only."""
    groups: dict[str, list[str]] = defaultdict(list)
    for m in members:
        desig = m.get("designation") or "Faculty"
        groups[desig].append(m["name"])

    if language == "ur":
        short_dept = "Computing" if "Computer Science" in dept else dept.replace("Department of ", "")
        lines = [f"Bilkul, IST {short_dept} ke faculty members yeh hain:"]
        for desig, names in sorted(groups.items(), key=lambda x: x[0]):
            lines.append(f"{desig}: {', '.join(names)}.")
        lines.append("Poori list official faculty directory se hai.")
        return " ".join(lines)

    lines = [f"Here are the faculty members of the {dept} at IST:"]
    for desig, names in sorted(groups.items(), key=lambda x: x[0]):
        lines.append(f"{desig}: {', '.join(names)}.")
    lines.append("This list is taken from the official IST faculty directory.")
    return " ".join(lines)


def get_department_members(dept: str) -> list[dict[str, str]]:
    return list(_by_dept.get(dept, []))


def try_faculty_answer(question: str, language: str) -> str | None:
    if not _by_dept:
        reload()
    if not is_faculty_list_query(question):
        return None
    dept = detect_department(question)
    if not dept:
        return None
    members = _by_dept.get(dept)
    if not members:
        for key, mems in _by_dept.items():
            if dept.lower() in key.lower() or key.lower() in dept.lower():
                members = mems
                dept = key
                break
    if not members:
        return None
    return format_faculty_answer(dept, members, language)
