"""
Deterministic required/preferred section split and conservative OR skill groups.

No LLM. Low-confidence patterns stay as independent singletons.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from app.processors.tokenizer import TextProcessor, canonicalize_skill, get_text_processor


PREFERRED_HEADERS = (
    "preferred qualifications",
    "preferred skills",
    "desired qualifications",
    "nice to have",
    "nice-to-have",
)

# "qualifications" is the common Greenhouse heading for the required block
# (Palo Alto and many others). Match as a full line only, never as a substring
# of "Preferred Qualifications".
REQUIRED_HEADERS = (
    "minimum qualifications",
    "required qualifications",
    "required skills",
    "basic qualifications",
    "requirements",
    "qualifications",
)

_HEADER_ALIASES = {h: "preferred" for h in PREFERRED_HEADERS}
_HEADER_ALIASES.update({h: "required" for h in REQUIRED_HEADERS})
# Longer headers first so "preferred qualifications" wins over "qualifications".
_HEADER_PATTERN = re.compile(
    r"^\s*(?P<header>"
    + "|".join(re.escape(h) for h in sorted(_HEADER_ALIASES, key=len, reverse=True))
    + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Slash run of tokens such as Go/C#/Java/C++. ci/cd is later rejected
# because its parts are not known skills.
_SLASH_RUN = re.compile(
    r"(?<![/\w+#])((?:[\w+#.+-]+)(?:\s*/\s*[\w+#.+-]+)+)(?![/\w+#])"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _known_canonicals() -> Set[str]:
    names = set(TextProcessor.TECH_SKILLS) | set(TextProcessor.SKILL_ALIASES.keys())
    return {canonicalize_skill(n) for n in names}


def known_skill(token: str) -> Optional[str]:
    """Return canonical name if token is a known skill, else None."""
    canon = canonicalize_skill(token)
    if not canon:
        return None
    if canon in _known_canonicals():
        return canon
    return None


def normalize_jd_text(text: Optional[str]) -> str:
    """Turn Greenhouse-style HTML into line-oriented plain text."""
    if not text:
        return ""
    out = text
    out = re.sub(r"(?i)<br\s*/?>", "\n", out)
    out = re.sub(r"(?i)</p>", "\n", out)
    out = re.sub(r"(?i)</div>", "\n", out)
    out = re.sub(r"(?i)</h[1-6]>", "\n", out)
    out = re.sub(r"(?i)<li[^>]*>", "\n", out)
    out = re.sub(r"<[^>]+>", "", out)
    out = html.unescape(out)
    out = out.replace("\xa0", " ")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


@dataclass
class SectionSplit:
    required_text: str
    preferred_text: str
    found_required_header: bool
    found_preferred_header: bool


def split_required_preferred(text: str) -> SectionSplit:
    """
    Split normalized JD text into required vs preferred slices.

    Preferred header alone is enough to exclude that slice from required.
    If a required header is also found, required skills come from that
    slice only (not from responsibilities/about fluff).
    """
    if not text:
        return SectionSplit("", "", False, False)

    matches = list(_HEADER_PATTERN.finditer(text))
    if not matches:
        return SectionSplit(text, "", False, False)

    found_required = False
    found_preferred = False
    preamble = text[: matches[0].start()].strip()
    required_parts: List[str] = []
    preferred_parts: List[str] = []

    for i, match in enumerate(matches):
        kind = _HEADER_ALIASES[match.group("header").lower()]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if kind == "preferred":
            found_preferred = True
            if body:
                preferred_parts.append(body)
        else:
            found_required = True
            if body:
                required_parts.append(body)

    if found_required:
        required_text = "\n".join(required_parts)
    else:
        required_text = preamble

    return SectionSplit(
        required_text=required_text,
        preferred_text="\n".join(preferred_parts),
        found_required_header=found_required,
        found_preferred_header=found_preferred,
    )


def parse_skill_groups(text: str) -> List[List[str]]:
    """
    Parse high-confidence ANY_OF groups, then leftover known skills as singletons.

    Accept:
      - slash runs of 2+ known skills (Go/C#/Java/C++)
      - 2-5 known skills in one sentence joined with 'or'
    Reject:
      - comma lists with no 'or'
      - lists longer than 5
      - items that are not known skills
      - ci/cd (parts are not known skills)
    """
    if not text:
        return []

    groups: List[List[str]] = []
    grouped: Set[str] = set()
    masked = text

    for match in _SLASH_RUN.finditer(text):
        parts = [p.strip() for p in re.split(r"\s*/\s*", match.group(1)) if p.strip()]
        known = []
        for part in parts:
            canon = known_skill(part)
            if canon:
                known.append(canon)
        if len(known) >= 2:
            groups.append(_unique(known))
            grouped.update(known)
            masked = masked.replace(match.group(1), " ", 1)

    for sentence in _SENTENCE_SPLIT.split(masked):
        sentence = sentence.strip()
        if not sentence or not re.search(r"\bor\b", sentence, re.IGNORECASE):
            continue
        found = _known_in_text(sentence)
        leftover = [s for s in found if s not in grouped]
        if 2 <= len(leftover) <= 5:
            groups.append(leftover)
            grouped.update(leftover)

    for skill in _known_in_text(masked):
        if skill not in grouped:
            groups.append([skill])
            grouped.add(skill)

    return _merge_duplicate_groups(groups)


def _known_in_text(text: str) -> List[str]:
    processor = get_text_processor()
    return [canonicalize_skill(s) for s in processor.extract_skills(text)]


def _unique(values: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _merge_duplicate_groups(groups: Sequence[Sequence[str]]) -> List[List[str]]:
    """Collapse restated copies of the same alternative set (slash + 'or' line)."""
    merged: List[List[str]] = []
    seen: Set[frozenset] = set()
    for group in groups:
        key = frozenset(group)
        if key in seen:
            continue
        seen.add(key)
        merged.append(list(group))
    return merged


def flatten_groups(groups: Iterable[Sequence[str]]) -> List[str]:
    return sorted({skill for group in groups for skill in group})
