"""
Extract canonical skills from raw text using:
  1. Exact alias matching against the skill taxonomy (word-boundary aware)
  2. Fuzzy matching (rapidfuzz) to catch typos / minor variants that aren't
     in the alias list (e.g. "Reactjs" vs "React js", "Mongo DB")
"""
import re
from rapidfuzz import fuzz
from skill_taxonomy import ALIAS_TO_CANONICAL, SOFT_ALIAS_TO_CANONICAL, get_all_canonical_skills


def _normalize(text: str) -> str:
    return text.lower()


def extract_skills_exact(text: str) -> set:
    """Find canonical skills via exact alias substring match with word boundaries."""
    text_norm = _normalize(text)
    found = set()
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        # word-boundary-ish match; allow aliases with punctuation like "c++", "node.js"
        pattern = re.escape(alias)
        if re.search(r"(?<![a-zA-Z0-9])" + pattern + r"(?![a-zA-Z0-9])", text_norm):
            found.add(canonical)
    return found


def extract_skills_fuzzy(text: str, existing: set, threshold: int = 87) -> set:
    """
    Catch near-miss variants of skills not caught by exact match, by fuzzy-comparing
    each word/bigram token in the text against known aliases.
    Only checks skills not already found, to save time.
    """
    text_norm = _normalize(text)
    tokens = re.findall(r"[a-zA-Z0-9+.#]+", text_norm)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    candidates = set(tokens) | set(bigrams)

    found = set()
    remaining_aliases = {a: c for a, c in ALIAS_TO_CANONICAL.items() if c not in existing}
    for alias, canonical in remaining_aliases.items():
        if len(alias) < 4:
            continue  # skip very short aliases for fuzzy (too noisy, e.g. "js", "ts", "db")
        for cand in candidates:
            if fuzz.ratio(alias, cand) >= threshold:
                found.add(canonical)
                break
    return found


def extract_skills(text: str, use_fuzzy: bool = True) -> set:
    """Full skill extraction pipeline: exact then fuzzy fallback."""
    exact = extract_skills_exact(text)
    if not use_fuzzy:
        return exact
    fuzzy = extract_skills_fuzzy(text, exact)
    return exact | fuzzy


def extract_soft_skills(text: str) -> set:
    """Extract soft skills (communication, teamwork, etc.) separately from technical skills.
    These are informational only and never affect the required-skill match ratio."""
    text_norm = _normalize(text)
    found = set()
    for alias, canonical in SOFT_ALIAS_TO_CANONICAL.items():
        pattern = re.escape(alias)
        if re.search(r"(?<![a-zA-Z0-9])" + pattern + r"(?![a-zA-Z0-9])", text_norm):
            found.add(canonical)
    return found


def extract_years_experience(text: str) -> float:
    """
    Best-effort extraction of total years of experience mentioned in text.
    Looks for patterns like "3 years", "2+ years of experience", etc.
    Returns 0 if nothing found (common for interns/freshers).
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of)?\s*experience", text.lower())
    if matches:
        return max(float(m) for m in matches)
    return 0.0
