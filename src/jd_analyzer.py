"""
Analyze a Job Description to separate:
  - required/must-have skills
  - preferred/nice-to-have skills
using section-header heuristics (e.g. "Requirements", "Preferred", "Nice to have").
Falls back to treating all extracted skills as required if no clear sections exist.
"""
import re
from skill_extractor import extract_skills

REQUIRED_HEADERS = [
    "requirements", "required skills", "must have", "must-have",
    "minimum qualifications", "qualifications", "what you need", "essential"
]
PREFERRED_HEADERS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "good to have",
    "preferred qualifications", "plus"
]
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(" + "|".join(REQUIRED_HEADERS + PREFERRED_HEADERS +
                         ["responsibilities", "about", "role", "benefits", "who you are"]) +
    r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE
)


def split_jd_sections(jd_text: str) -> dict:
    """Split JD text into named sections based on header lines."""
    lines = jd_text.split("\n")
    sections = {"_preamble": []}
    current = "_preamble"
    for line in lines:
        header_match = re.match(
            r"^\s*(" + "|".join(REQUIRED_HEADERS + PREFERRED_HEADERS +
                                 ["responsibilities", "about", "role", "benefits", "who you are"]) +
            r")\b.*$",
            line, re.IGNORECASE
        )
        if header_match and len(line.strip()) < 60:
            current = header_match.group(1).lower()
            sections[current] = []
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
        lname = section_name.lower()
        if any(h in lname for h in REQUIRED_HEADERS):
            required_text_parts.append(content)
        elif any(h in lname for h in PREFERRED_HEADERS):
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
