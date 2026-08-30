"""
Structure detection for HSG150 (Health and safety in construction).

Approach B: numbered paragraphs are the chunk boundaries. Section and
topic are forward-filled from the table of contents. Subheadings
("Tower scaffolds", "Washing facilities") are metadata attached to the
paragraphs they precede -- they do not start a new chunk by themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_assistant.headings import normalise_heading

# Top-level sections in publication order.
HSG150_SECTIONS: list[tuple[str, str]] = [
    ("1", "Section 1: Preparing for work"),
    ("2", "Section 2: Setting up the site"),
    ("3", "Section 3: Construction-phase health and safety"),
    ("4", "Section 4: Health and safety management and the law"),
]

# Topic titles from the table of contents, in document order.
HSG150_TOPICS: list[str] = [
    "Planning the work",
    "Organising the work",
    "Notifying the site to HSE",
    "Site access",
    "Site boundaries",
    "Welfare facilities",
    "Good order, storage areas and waste materials",
    "Lighting",
    "Emergency procedures",
    "Fire",
    "First aid",
    "Reporting injuries, diseases and dangerous occurrences",
    "Site rules",
    "Site management and supervision",
    "Working at height",
    "Site traffic and mobile plant",
    "Moving goods safely",
    "Groundwork",
    "Demolition, dismantling and structural alteration",
    "Occupational health risks",
    "Electricity",
    "Slips and trips",
    "Working in confined spaces",
    "Prevention of drowning",
    "Protective equipment",
    "Work affecting the public",
    "Monitoring and reviewing",
]

# Titles that may bleed onto the end of a paragraph when the next heading
# sits above the following paragraph number. Compared by normalised form,
# so "Section 3: Construction-phase health and safety" also matches the
# extracted variant "3: Construction-phase health and safety".
HSG150_KNOWN_HEADINGS: frozenset[str] = frozenset(
    normalise_heading(title)
    for title in [label for _code, label in HSG150_SECTIONS] + HSG150_TOPICS
)

SECTION_START_RE = re.compile(
    r"(?m)^(?:(\d):\s+(Preparing for work|Setting up the site|"
    r"Construction-phase health and(?:\s+safety)?|"
    r"Health and safety(?:\s+management and the law)?)|"
    r"Section\s+(\d)\s*:\s*(.+))$"
)

FIGURE_OR_TABLE_RE = re.compile(r"^(?:Figure|Table)\b", re.I)
PARAGRAPH_START_RE = re.compile(r"^\d{1,3}\s+(?=[A-Z])")

# A subheading is a short Title-Case (or similar) phrase, not a paragraph
# number and not a figure/table caption. Matched as a whole line, or as a
# glued tail at the end of a paragraph (same pattern as HSG141 case studies).
SUBHEADING_LINE_RE = re.compile(
    r"(?m)^(?!Figure\b)(?!Table\b)(?!\d{1,3}\s)([A-Z][^\n]{2,60})$"
)
GLUED_SUBHEADING_RE = re.compile(
    r"([.!?])\s+((?:[A-Z][a-z]+(?:[\s-][A-Za-z][a-z']*){0,7}))\s*$"
)


@dataclass
class ParagraphLabel:
    """Structure labels for one numbered paragraph."""

    section: str
    topic: str
    subtopic: str


def _section_label(number: str, title: str = "") -> str:
    for code, label in HSG150_SECTIONS:
        if code == number:
            return label
    if title:
        return f"Section {number}: {title.strip()}"
    return f"Section {number}"


def find_section_starts(text: str) -> list[tuple[int, str]]:
    """Return (offset, section_label) for each section heading in order."""
    found: list[tuple[int, str]] = []
    for match in SECTION_START_RE.finditer(text):
        if match.group(1):
            found.append((match.start(), _section_label(match.group(1))))
        else:
            found.append((match.start(), _section_label(match.group(3), match.group(4))))
    # Also match compact forms used as body headings: "1: Preparing for work"
    compact = re.compile(
        r"(?m)^(1:\s+Preparing for work|2:\s+Setting up the site|"
        r"3:\s+Construction-phase health and(?:\s+safety)?|"
        r"4:\s+Health and safety(?:\s+management and the law)?)\s*$"
    )
    for match in compact.finditer(text):
        code = match.group(1)[0]
        label = _section_label(code)
        if not any(abs(offset - match.start()) < 5 and label == existing for offset, existing in found):
            found.append((match.start(), label))
    found.sort(key=lambda item: item[0])
    return found


def find_topic_starts(text: str) -> list[tuple[int, str]]:
    """Return (offset, topic_title) for each ToC topic heading."""
    found: list[tuple[int, str]] = []
    for topic in HSG150_TOPICS:
        # Prefer a whole-line match; fall back to a line that starts with the topic.
        pattern = re.compile(
            r"(?m)^" + re.escape(topic) + r"(?:\s*$|(?=\s*\n))",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            found.append((match.start(), topic))
    found.sort(key=lambda item: item[0])
    # Deduplicate overlapping matches (keep first at each offset).
    deduped: list[tuple[int, str]] = []
    for offset, topic in found:
        if deduped and abs(deduped[-1][0] - offset) < 3:
            continue
        deduped.append((offset, topic))
    return deduped


def _looks_like_subheading(candidate: str, standalone: bool = False) -> bool:
    line = candidate.strip()
    if not line or len(line) > 60:
        return False
    if FIGURE_OR_TABLE_RE.match(line):
        return False
    if PARAGRAPH_START_RE.match(line):
        return False
    if line.lower() in {"introduction", "contents", "foreword", "yes", "no"}:
        return False
    # Known topics are section-level, not subtopics.
    if line in HSG150_TOPICS:
        return False
    if re.match(r"^\d:\s+", line):
        return False
    if re.match(r"^Section\s+\d", line, re.I):
        return False
    words = line.split()
    if not (1 <= len(words) <= 10):
        return False
    if not line[0].isupper():
        return False
    if line[-1] in ".?!:;":
        return False
    # A line of its own is already strong evidence, and HSE sets many
    # headings in sentence case ("Health and safety competence"). A tail
    # glued onto a paragraph is far more likely to be body text, so it
    # still has to look Title Case to qualify.
    if standalone:
        return True
    caps = sum(1 for word in words if word[:1].isupper())
    return caps >= max(1, len(words) // 2)


def extract_subheading_from_gap(gap_text: str) -> str:
    """
    Find a subheading in the text between two numbered paragraphs.

    Searches whole lines first, then a glued Title-Case tail (the same
    pattern that left "Washing facilities" stuck to the end of para 50).
    """
    if not gap_text or not gap_text.strip():
        return ""

    # Last matching line in the gap wins: that is the heading immediately
    # before the next paragraph.
    candidates = [
        match.group(1).strip()
        for match in SUBHEADING_LINE_RE.finditer(gap_text)
        if _looks_like_subheading(match.group(1), standalone=True)
    ]
    if candidates:
        return candidates[-1]

    glued = GLUED_SUBHEADING_RE.search(gap_text.strip())
    if glued and _looks_like_subheading(glued.group(2)):
        return glued.group(2).strip()
    return ""


def peel_trailing_subheading(para_text: str) -> tuple[str, str]:
    """
    If a paragraph ends with a glued subheading for the *next* section,
    strip it from display text and return (cleaned_text, subheading).
    """
    lines = para_text.rstrip().split("\n")
    if len(lines) >= 2 and _looks_like_subheading(lines[-1], standalone=True):
        return "\n".join(lines[:-1]).strip(), lines[-1].strip()

    glued = GLUED_SUBHEADING_RE.search(para_text.rstrip())
    if glued and _looks_like_subheading(glued.group(2)):
        cleaned = para_text[: glued.start(2)].rstrip()
        return cleaned, glued.group(2).strip()
    return para_text.strip(), ""


def _label_at(offset: int, starts: list[tuple[int, str]], default: str = "") -> str:
    current = default
    for start, label in starts:
        if start <= offset:
            current = label
        else:
            break
    return current


def label_paragraphs(
    text: str, paragraph_starts: list[tuple[int, int]]
) -> list[ParagraphLabel]:
    """
    Attach section / topic / subtopic to each (para_number, start_offset).

    Subtopics are taken from the gap before each paragraph (or peeled from
    the previous paragraph's trailing glued heading) and forward-filled
    until the next subheading or a topic change.
    """
    section_starts = find_section_starts(text)
    topic_starts = find_topic_starts(text)

    labels: list[ParagraphLabel] = []
    current_subtopic = ""
    pending_subtopic = ""

    for index, (para_num, start) in enumerate(paragraph_starts):
        end = (
            paragraph_starts[index + 1][1]
            if index + 1 < len(paragraph_starts)
            else len(text)
        )
        # Gap before this paragraph (after previous para start content).
        gap_start = 0 if index == 0 else paragraph_starts[index - 1][1]
        # Use only the interstitial after the previous paragraph's body
        # roughly: from previous start we don't re-scan the whole previous
        # body for headings -- peel handles trailing glue; gap from
        # previous end is approximated by text between starts minus the
        # previous number line. Simpler: scan text[prev_end:start] where
        # prev_end is previous paragraph's start... actually the gap is
        # everything from end of previous paragraph body. We approximate
        # with text between paragraph starts excluding the previous body
        # by taking a short window before `start`.
        window = text[max(gap_start, start - 400) : start]
        gap_sub = extract_subheading_from_gap(window)
        if pending_subtopic:
            current_subtopic = pending_subtopic
            pending_subtopic = ""
        elif gap_sub:
            current_subtopic = gap_sub

        section = _label_at(start, section_starts, "Introduction")
        topic = _label_at(start, topic_starts, "")

        # Topic change clears a stale subtopic from the previous topic.
        if labels and labels[-1].topic and topic and topic != labels[-1].topic:
            if not gap_sub and not pending_subtopic:
                current_subtopic = ""

        # Peek: if this paragraph's body ends with a glued subheading,
        # that label belongs to the *next* paragraph.
        body = text[start:end]
        _, trailing = peel_trailing_subheading(body)
        if trailing:
            pending_subtopic = trailing

        labels.append(
            ParagraphLabel(section=section, topic=topic, subtopic=current_subtopic)
        )

    return labels
