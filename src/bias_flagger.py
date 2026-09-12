"""
Rule-based JD bias / overly-narrow-phrasing flagger.

Bonus feature from the problem statement: "Flag potential bias or overly
narrow phrasing in the JD itself that could unfairly exclude qualified
candidates."

Every flag below comes from a concrete, inspectable rule (keyword list,
regex, or a count compared against a threshold) — never an LLM judgment —
so a judge (or a recruiter) can see exactly why a flag fired, same
philosophy as the rest of the matching engine.

Flags produced (each a dict with type / severity / message / evidence):
  - gendered_language           masculine- or feminine-coded word skew
  - age_coded_language          phrasing that signals a preference for age
  - national_origin_language    phrasing that signals a nationality/language/accent preference
  - affinity_bias               "culture fit" language and elite-pedigree/prestige phrasing
  - seniority_experience_mismatch   "junior/intern" role demanding senior-level years
  - degree_requirement          hard degree requirement with no "or equivalent" clause
  - kitchen_sink_requirements   an unusually long *required* (not preferred) skill list
  - vendor_lock_in              a single cloud vendor pinned as required with no
                                 "or equivalent platform" allowance

Category taxonomy cross-checked against a labeled JD bias-audit dataset
(HuggingFace: 0xkamal7/hr-jd-bias-audit, categories: affinity / age / gender
/ national_origin) — added national_origin_language and affinity_bias here
after confirming those two categories weren't covered by any existing check.
"""
import re
from skill_extractor import extract_years_experience

# Word lists adapted from published research on gendered job-ad language
# (e.g. Gaucher, Friesen & Kay, 2011) — presence of a *skew* toward one list,
# not just presence of any single word, is what gets flagged.
MASCULINE_CODED = [
    "ninja", "rockstar", "rock star", "guru", "hacker", "dominant",
    "aggressive", "competitive", "fearless", "superior", "assertive",
    "driven", "ambitious", "independent", "analytical", "decisive",
]
FEMININE_CODED = [
    "collaborative", "supportive", "nurturing", "compassionate",
    "interpersonal", "dependable", "loyal", "cooperative", "warm",
    "considerate", "committed to our team",
]

# Single standalone words fire on their own (a bare "young" describing a
# desired candidate trait is already a strong, explicit signal); multi-word
# phrases need the full phrase to match.
AGE_CODED = [
    "digital native", "young and energetic", "recent graduate only",
    "youthful", "fresh graduate only", "energetic young team",
    "young", "young, energetic", "young, ambitious",
]

# Nationality / language / accent preference. Deliberately narrow — things
# like "authorized to work in the US" or "must have a valid work visa" are
# legitimate legal/logistics constraints, not bias, so they're excluded.
NATIONAL_ORIGIN_CODED = [
    "native english speaker", "native speaker of english",
    "native-level english", "no accent", "must speak english as first language",
    "local candidates only", "must be a local", "us-born",
]

# "Affinity bias" — favoring candidates who resemble the existing team,
# expressed either as vague culture-fit language or as a preference for
# elite/prestigious pedigree rather than demonstrated skill.
AFFINITY_CODED = [
    "cultural fit", "culturally aligned", "culture fit", "fit right in",
    "like family", "our way of doing things", "work hard play hard",
    "great cultural fit",
]
PEDIGREE_CODED = [
    "top university", "top-tier university", "prestigious institution",
    "prestigious university", "ivy league", "elite university",
    "elite institution", "top-tier institution",
]

SENIORITY_HINTS_JUNIOR = [
    "junior", "intern", "internship", "entry level", "entry-level",
    "fresher", "trainee",
]
SENIORITY_HINTS_SENIOR = [
    "senior", "lead", "principal", "staff engineer", "architect",
    "engineering manager", "head of",
]

CLOUD_SKILLS = {"aws", "azure", "gcp"}


def _find_hits(text: str, words: list) -> list:
    text_l = text.lower()
    hits = []
    for w in words:
        pattern = r"(?<![a-zA-Z])" + re.escape(w.lower()) + r"(?![a-zA-Z])"
        if re.search(pattern, text_l):
            hits.append(w)
    return hits


def _flag_gendered_language(jd_text: str) -> list:
    masc = _find_hits(jd_text, MASCULINE_CODED)
    fem = _find_hits(jd_text, FEMININE_CODED)
    flags = []
    if len(masc) >= 2 and len(masc) > len(fem):
        flags.append({
            "type": "gendered_language",
            "severity": "medium",
            "message": (
                f"The JD leans on masculine-coded language ({', '.join(masc)}). "
                "Research on job-ad wording links a heavy skew toward words like "
                "these to fewer women applying, even when the role itself is neutral."
            ),
            "evidence": masc,
        })
    elif len(fem) >= 2 and len(fem) > len(masc):
        flags.append({
            "type": "gendered_language",
            "severity": "low",
            "message": (
                f"The JD leans on feminine-coded language ({', '.join(fem)}). "
                "A heavy skew toward words like these has been linked in research "
                "to fewer men applying."
            ),
            "evidence": fem,
        })
    return flags


def _flag_age_coded(jd_text: str) -> list:
    hits = set(_find_hits(jd_text, AGE_CODED))
    if not hits:
        return []
    return [{
        "type": "age_coded_language",
        "severity": "high",
        "message": (
            f"Phrasing like {', '.join(sorted(hits))} signals an age preference. This is "
            "both a fairness concern and, in many jurisdictions, a legal risk — "
            "consider describing the trait actually needed (e.g. 'comfortable "
            "learning new tools quickly') instead."
        ),
        "evidence": sorted(hits),
    }]


def _flag_national_origin(jd_text: str) -> list:
    hits = _find_hits(jd_text, NATIONAL_ORIGIN_CODED)
    if not hits:
        return []
    return [{
        "type": "national_origin_language",
        "severity": "high",
        "message": (
            f"Phrasing like {', '.join(hits)} signals a nationality, language, or "
            "accent preference unrelated to the job itself, which risks unfairly "
            "excluding qualified candidates and is a legal risk in many "
            "jurisdictions. If strong English communication is genuinely required, "
            "describe the required proficiency level directly instead (e.g. "
            "'able to communicate clearly with the team in English')."
        ),
        "evidence": hits,
    }]


def _flag_affinity_bias(jd_text: str) -> list:
    culture_hits = _find_hits(jd_text, AFFINITY_CODED)
    pedigree_hits = _find_hits(jd_text, PEDIGREE_CODED)
    hits = culture_hits + pedigree_hits
    if not hits:
        return []
    detail = []
    if culture_hits:
        detail.append(
            f"vague 'culture fit' language ({', '.join(culture_hits)}) tends to "
            "favor candidates who resemble the existing team rather than "
            "candidates who can do the job"
        )
    if pedigree_hits:
        detail.append(
            f"a preference for elite/prestigious pedigree ({', '.join(pedigree_hits)}) "
            "correlates with socioeconomic background more than with skill"
        )
    return [{
        "type": "affinity_bias",
        "severity": "medium",
        "message": (
            "The JD shows signs of affinity bias — " + "; and ".join(detail) + ". "
            "Consider replacing this with specific, job-relevant behaviors or "
            "outcomes you're actually looking for."
        ),
        "evidence": hits,
    }]


def _flag_seniority_experience_mismatch(jd_text: str) -> list:
    text_l = jd_text.lower()
    years = extract_years_experience(jd_text)
    is_junior_titled = any(h in text_l for h in SENIORITY_HINTS_JUNIOR)
    is_senior_titled = any(h in text_l for h in SENIORITY_HINTS_SENIOR)
    if is_junior_titled and not is_senior_titled and years >= 3:
        return [{
            "type": "seniority_experience_mismatch",
            "severity": "high",
            "message": (
                f"The role is framed as junior/intern-level but the JD asks for "
                f"{years:.0f}+ years of experience. That contradiction likely "
                "excludes exactly the entry-level candidates the role is meant for."
            ),
            "evidence": [f"{years:.0f}+ years experience required"],
        }]
    return []


def _flag_degree_requirement(jd_text: str) -> list:
    text_l = jd_text.lower()
    degree_match = re.search(
        r"bachelor'?s?\s+degree|master'?s?\s+degree|\bb\.?\s?tech\b|\bb\.e\.(?!\w)",
        text_l,
    )
    if not degree_match:
        return []
    has_equivalent_clause = re.search(
        r"or\s+equivalent(\s+(practical\s+)?experience)?", text_l
    )
    if has_equivalent_clause:
        return []
    is_junior = any(h in text_l for h in SENIORITY_HINTS_JUNIOR)
    return [{
        "type": "degree_requirement",
        "severity": "low" if is_junior else "medium",
        "message": (
            "A specific degree is required with no 'or equivalent experience' "
            "allowance. This can exclude self-taught, bootcamp-trained, or "
            "otherwise qualified candidates without that exact credential."
        ),
        "evidence": [degree_match.group(0)],
    }]


def _flag_kitchen_sink(jd_analysis: dict, threshold: int = 10) -> list:
    required = jd_analysis["required_skills"]
    if len(required) <= threshold:
        return []
    return [{
        "type": "kitchen_sink_requirements",
        "severity": "medium",
        "message": (
            f"{len(required)} distinct skills are marked as *required* (not "
            "preferred). Long must-have lists are known to discourage otherwise "
            "qualified candidates — particularly women, per published survey "
            "data — from applying if they don't check every single box. "
            "Consider moving some of these into a 'preferred' section."
        ),
        "evidence": sorted(required),
    }]


def _flag_vendor_lock_in(jd_text: str, jd_analysis: dict) -> list:
    present = CLOUD_SKILLS & jd_analysis["required_skills"]
    if len(present) != 1:
        return []
    vendor = next(iter(present)).upper()
    text_l = jd_text.lower()
    if re.search(r"or\s+(similar|equivalent|another|other)\s*(cloud|platform)?", text_l):
        return []
    return [{
        "type": "vendor_lock_in",
        "severity": "low",
        "message": (
            f"{vendor} is required specifically, rather than 'cloud platform "
            "experience (AWS/GCP/Azure)'. Cloud skills transfer readily between "
            "providers, so pinning one vendor as required may needlessly exclude "
            "candidates with equivalent experience elsewhere."
        ),
        "evidence": [vendor],
    }]


def analyze_jd_bias(jd_text: str, jd_analysis: dict) -> list:
    """
    Run all bias/narrow-phrasing checks against a JD.

    `jd_text` is the raw JD text; `jd_analysis` is the dict returned by
    jd_analyzer.analyze_jd() (needs 'required_skills').

    Returns a list of flag dicts: {type, severity, message, evidence}.
    Empty list means no flags were raised, not that the JD is guaranteed
    bias-free — this is a heuristic first-pass, not a legal or HR audit.
    """
    flags = []
    flags.extend(_flag_gendered_language(jd_text))
    flags.extend(_flag_age_coded(jd_text))
    flags.extend(_flag_national_origin(jd_text))
    flags.extend(_flag_affinity_bias(jd_text))
    flags.extend(_flag_seniority_experience_mismatch(jd_text))
    flags.extend(_flag_degree_requirement(jd_text))
    flags.extend(_flag_kitchen_sink(jd_analysis))
    flags.extend(_flag_vendor_lock_in(jd_text, jd_analysis))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: severity_order.get(f["severity"], 3))
    return flags
