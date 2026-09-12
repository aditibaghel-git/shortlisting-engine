"""
Analyze a Job Description to separate:
  - required/must-have skills
  - preferred/nice-to-have skills
using section-header heuristics (e.g. "Requirements", "Preferred", "Nice to have").
Falls back to treating all extracted skills as required if no clear sections exist.
"""
import re
from skill_extractor import extract_skills

# NOTE: header phrases are matched *separator-agnostically* (see _phrase_to_pattern
# below), so "must have" also matches "MUST-HAVE", "must_have", "musthave", etc.
# We still list a few explicit variants (e.g. plurals) for clarity/robustness.
REQUIRED_HEADERS = [
    "requirements", "required skills", "required qualifications",
    "must have", "must haves", "minimum qualifications", "qualifications",
    "what you need", "essential", "essential skills", "technical skills",
    "core skills", "key skills",
]
PREFERRED_HEADERS = [
    "preferred", "preferred skills", "nice to have", "nice to haves",
    "bonus", "good to have", "good to haves", "preferred qualifications",
    "plus", "desired skills", "optional skills",
]
# Recognized headers that are neither required nor preferred (e.g. soft
# skills, responsibilities). These still need to be recognized as section
# boundaries so their content doesn't get swallowed into whichever
# required/preferred section came before them.
OTHER_HEADERS = [
    "responsibilities", "about", "role", "benefits", "who you are",
    "soft skills", "soft skill", "skills", "about the role", "about us",
]

ALL_HEADERS = REQUIRED_HEADERS + PREFERRED_HEADERS + OTHER_HEADERS


def _norm_phrase(s: str) -> str:
    """Lowercase and collapse hyphens/underscores/slashes/whitespace to a
    single space, so 'GOOD-TO-HAVE', 'good_to_have' and 'good to have' all
    normalize to the same key."""
    return re.sub(r"[\s\-_/]+", " ", s.strip().lower())


def _phrase_to_pattern(phrase: str) -> str:
    """Turn a normalized header phrase into a regex that matches it with
    any mix of spaces/hyphens/underscores/slashes between words."""
    words = _norm_phrase(phrase).split(" ")
    return r"[\s\-_/]+".join(re.escape(w) for w in words)


REQUIRED_HEADERS_NORM = {_norm_phrase(h) for h in REQUIRED_HEADERS}
PREFERRED_HEADERS_NORM = {_norm_phrase(h) for h in PREFERRED_HEADERS}

# Longest phrases first so e.g. "required skills" wins over a shorter
# alternative that happens to be a prefix of it.
_HEADER_PATTERNS = sorted(
    {_norm_phrase(h) for h in ALL_HEADERS}, key=len, reverse=True
)
HEADER_REGEX = re.compile(
    r"^\s*(" + "|".join(_phrase_to_pattern(h) for h in _HEADER_PATTERNS) + r")\b.*$",
    re.IGNORECASE,
)


def split_jd_sections(jd_text: str) -> dict:
    """Split JD text into named sections based on header lines.

    A line is treated as a new section header if it *starts* with one of the
    known header phrases (separator-agnostic) and is short (looks like a
    header, not a sentence). Sections are keyed by the normalized header
    phrase so callers can classify them consistently.
    """
    lines = jd_text.split("\n")
    sections = {"_preamble": []}
    current = "_preamble"
    for line in lines:
        header_match = HEADER_REGEX.match(line)
        if header_match and len(line.strip()) < 60:
            current = _norm_phrase(header_match.group(1))
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def analyze_jd(jd_text: str) -> dict:
    """
    Returns:
      {
        'required_skills': set,
        'preferred_skills': set,
        'all_skills': set,
        'full_text': str
      }
    """
    sections = split_jd_sections(jd_text)

    required_text_parts = []
    preferred_text_parts = []
    other_text_parts = []

    for section_name, content in sections.items():
        # section_name is already a normalized header phrase (see
        # split_jd_sections), so this is an exact-set check, not a fragile
        # substring check that could mis-tag e.g. a "role" section as
        # "preferred" just because "role" was substringy with something.
        if section_name in REQUIRED_HEADERS_NORM:
            required_text_parts.append(content)
        elif section_name in PREFERRED_HEADERS_NORM:
            preferred_text_parts.append(content)
        else:
            other_text_parts.append(content)

    required_text = "\n".join(required_text_parts)
    preferred_text = "\n".join(preferred_text_parts)
    other_text = "\n".join(other_text_parts)

    if required_text.strip():
        required_skills = extract_skills(required_text)
    else:
        # No explicit "requirements" section found -> treat all skills in JD as required
        required_skills = extract_skills(jd_text)

    preferred_skills = extract_skills(preferred_text) if preferred_text.strip() else set()
    # Remove overlap: a skill can't be both required and preferred
    preferred_skills -= required_skills

    all_skills = required_skills | preferred_skills | extract_skills(other_text)

    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "all_skills": all_skills,
        "full_text": jd_text,
    }
