"""
Streamlit demo UI for the Smart Shortlisting Engine — redesigned, no sidebar.

Run with:
    streamlit run app.py
"""
import os
import sys
import tempfile
import zipfile
import re
import xml.etree.ElementTree as ET
import textwrap

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))

from parser import extract_text, extract_candidate_name
from jd_analyzer import analyze_jd
from skill_extractor import extract_skills, extract_soft_skills
from matcher import match_all_resumes_to_jd
from explainer import generate_explanation, format_skill_list
from bias_flagger import analyze_jd_bias

st.set_page_config(page_title="Shortlisting Engine", page_icon="◆", layout="wide")

# ----------------------------------------------------------------------------
# Design tokens / global CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #E7ECEF;
    --ink-soft: #9AA5AC;
    --bg: #0F1417;
    --card: #171D21;
    --line: #2A3238;
    --teal: #3FC2B0;
    --teal-soft: rgba(63,194,176,0.14);
    --coral: #E2705A;
    --coral-soft: rgba(226,112,90,0.14);
    --gold: #D9AE55;
    --gold-soft: rgba(217,174,85,0.14);
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }

h1, h2, h3, .hero-title { font-family: 'Lora', serif; color: var(--ink); }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1080px; }

/* Hero */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-sub {
    color: var(--ink-soft);
    font-size: 1.05rem;
    max-width: 640px;
    margin-bottom: 0.4rem;
}
.formula-chip {
    display: inline-block;
    font-family: 'Inter', monospace;
    font-size: 0.82rem;
    background: var(--teal-soft);
    color: var(--teal);
    border: 1px solid rgba(31,122,108,0.25);
    padding: 5px 12px;
    border-radius: 6px;
    margin-top: 0.6rem;
}

/* Upload cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border-color: var(--line);
    border-radius: 10px;
    padding: 0.35rem 0.7rem 0.2rem 0.7rem;
}
.upload-card-label {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.1rem;
}
.upload-card-hint {
    color: var(--ink-soft);
    font-size: 0.82rem;
    margin-bottom: 0.7rem;
}

/* Section headers */
.section-label {
    font-family: 'Lora', serif;
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 2.2rem;
    margin-bottom: 0.9rem;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.5rem;
}

/* Skill pills */
.pill {
    display: inline-block;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 2px 4px 2px 0;
    font-weight: 500;
}
.pill-required { background: var(--teal-soft); color: var(--teal); }
.pill-preferred { background: var(--gold-soft); color: var(--gold); }

/* Bias flag cards */
.bias-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 4px solid var(--line);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.bias-card.sev-high { border-left-color: var(--coral); }
.bias-card.sev-medium { border-left-color: var(--gold); }
.bias-card.sev-low { border-left-color: var(--teal); }
.bias-sev-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 999px;
    margin-right: 8px;
}
.bias-sev-tag.sev-high { background: var(--coral-soft); color: var(--coral); }
.bias-sev-tag.sev-medium { background: var(--gold-soft); color: var(--gold); }
.bias-sev-tag.sev-low { background: var(--teal-soft); color: var(--teal); }
.bias-msg { font-size: 0.88rem; line-height: 1.5; margin-top: 6px; }
.bias-evidence { color: var(--ink-soft); font-size: 0.78rem; margin-top: 4px; }

/* Ranked candidate row */
.rank-row {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
}
.rank-row.top1 { border-left: 4px solid var(--teal); }
.rank-badge {
    font-family: 'Lora', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--ink-soft);
    min-width: 34px;
}
.rank-row.top1 .rank-badge { color: var(--teal); }
.rank-name { font-weight: 600; font-size: 1.02rem; }
.rank-meta { color: var(--ink-soft); font-size: 0.8rem; margin-top: 2px; }
.rank-score {
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--ink);
    min-width: 70px;
    text-align: right;
}
.subscore-bar-track {
    background: var(--line);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.subscore-bar-fill { height: 100%; border-radius: 4px; }

/* Explanation cards */
.explain-card {
    background: var(--card);
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
}
.explain-rank-tag {
    font-family: 'Lora', serif;
    font-weight: 600;
    color: var(--teal);
    font-size: 0.95rem;
}
.explain-score {
    float: right;
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 1.15rem;
}
.explain-row { margin: 0.45rem 0; font-size: 0.92rem; line-height: 1.5; }
.tag-ok { color: var(--teal); font-weight: 600; }
.tag-missing { color: var(--coral); font-weight: 600; }
.verdict-strong { color: var(--teal); font-weight: 600; }
.verdict-partial { color: var(--gold); font-weight: 600; }
.verdict-weak { color: var(--coral); font-weight: 600; }

.stButton>button {
    background: var(--teal);
    color: #0F1417;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}
.stButton>button:hover { background: #55D4C2; color: #0F1417; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">Smart Shortlisting Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Rank a batch of resumes against a job description using explicit '
    'skill matching and contextual semantic similarity — every score is fully explainable, '
    'no black-box LLM judgment involved.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="formula-chip">final = 100 × (0.5 × keyword + 0.5 × semantic) — '
    'or, when a historical resume corpus is available, '
    '100 × (0.50 × keyword + 0.25 × semantic + 0.25 × learned)</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Upload section
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">1 · Upload</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown('<div class="upload-card-label">Job Description</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-card-hint">One PDF or TXT file</div>', unsafe_allow_html=True)
        jd_file = st.file_uploader(
            "Upload Job Description", type=["pdf", "txt"], key="jd",
            label_visibility="collapsed",
        )

with col2:
    with st.container(border=True):
        st.markdown('<div class="upload-card-label">Resumes</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-card-hint">Select the full batch — PDF or TXT</div>', unsafe_allow_html=True)
        resume_files = st.file_uploader(
            "Upload Resumes", type=["pdf", "txt", "docx"], accept_multiple_files=True, key="resumes",
            label_visibility="collapsed",
        )

st.write("")
run_col, _ = st.columns([1, 3])
with run_col:
    run_button = st.button("Run Matching →", type="primary", use_container_width=True)


def _extract_docx_text(filepath):
    """Extract paragraph/table text from a DOCX without requiring another parser module."""
    try:
        with zipfile.ZipFile(filepath) as z:
            xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        parts = []
        for para in root.findall(".//w:p", ns):
            text = "".join((node.text or "") for node in para.findall(".//w:t", ns))
            if text.strip():
                parts.append(text)
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_xml_text(filepath):
    """Extract readable text from XML-like resume files."""
    try:
        root = ET.parse(filepath).getroot()
        return "\n".join(t.strip() for t in root.itertext() if t and t.strip())
    except Exception:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return re.sub(r"<[^>]+>", " ", f.read())


def extract_training_text(filepath):
    """Read supported historical resume formats for the learned corpus."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".docx":
        raw = _extract_docx_text(filepath)
    elif ext == ".xml":
        raw = _extract_xml_text(filepath)
    else:
        raw = extract_text(filepath)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def find_training_zip():
    """Find the historical resume ZIP without requiring another code file."""
    base = os.path.dirname(__file__)
    candidates = [
        os.environ.get("TRAINING_RESUMES_ZIP", ""),
        os.path.join(base, "training_resumes.zip"),
        os.path.join(base, "data", "training_resumes.zip"),
        os.path.join(base, "Dummy Resumes-20260912T080512Z-1-001.zip"),
        os.path.join(base, "data", "Dummy Resumes-20260912T080512Z-1-001.zip"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


@st.cache_resource(show_spinner=False)
def train_resume_corpus(zip_path, modified_time):
    """Fit a TF-IDF representation on historical resumes.

    This is unsupervised corpus training: it learns vocabulary/IDF statistics
    from previous resumes, not recruiter selection labels.
    """
    texts = []
    names = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                ext = os.path.splitext(member.filename)[1].lower()
                if ext not in {".pdf", ".txt", ".docx", ".xml"}:
                    continue
                safe_name = os.path.basename(member.filename)
                if not safe_name:
                    continue
                path = os.path.join(tmpdir, safe_name)
                with open(path, "wb") as f:
                    f.write(z.read(member))
                try:
                    text = extract_training_text(path)
                except Exception:
                    text = ""
                if len(text.split()) >= 5:
                    texts.append(text)
                    names.append(member.filename)

    if not texts:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1,
        max_df=0.98, sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return None
    return {"vectorizer": vectorizer, "matrix": matrix, "count": len(texts), "names": names}


def learned_semantic_scores(jd_text, resume_texts, model):
    """Score new resumes against the JD in the vocabulary learned from history."""
    if not model or not resume_texts:
        return [None] * len(resume_texts)
    try:
        v = model["vectorizer"]
        jd_vec = v.transform([jd_text])
        resume_vecs = v.transform(resume_texts)
        raw = cosine_similarity(jd_vec, resume_vecs)[0]
        lo, hi = float(raw.min()), float(raw.max())
        if hi - lo < 1e-9:
            return raw.tolist()
        return ((raw - lo) / (hi - lo)).tolist()
    except Exception:
        return [None] * len(resume_texts)


def save_uploaded_file(uploaded_file, tmpdir):
    path = os.path.join(tmpdir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def score_color(score_pct):
    """0-100 -> css color var for bar fills."""
    if score_pct >= 65:
        return "var(--teal)"
    elif score_pct >= 35:
        return "var(--gold)"
    return "var(--coral)"


def verdict_class(req_ratio, score):
    if req_ratio == 1.0 and score >= 70:
        return "verdict-strong", "Strong fit"
    elif req_ratio >= 0.5:
        return "verdict-partial", "Partial fit"
    return "verdict-weak", "Weak fit"


if run_button:
    if not jd_file:
        st.error("Please upload a Job Description first.")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Parsing resumes and running keyword + semantic matching..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                jd_path = save_uploaded_file(jd_file, tmpdir)
                jd_text = extract_text(jd_path)
                jd_analysis = analyze_jd(jd_text)
                jd_bias_flags = analyze_jd_bias(jd_text, jd_analysis)

                training_zip = find_training_zip()
                training_model = None
                if training_zip:
                    training_model = train_resume_corpus(training_zip, os.path.getmtime(training_zip))

                resumes = []
                for rf in resume_files:
                    rpath = save_uploaded_file(rf, tmpdir)
                    ext = os.path.splitext(rf.name)[1].lower()
                    if ext == ".docx":
                        text = extract_training_text(rpath)
                    else:
                        text = extract_text(rpath)
                    name = extract_candidate_name(text, fallback=os.path.splitext(rf.name)[0])
                    skills = extract_skills(text)
                    soft_skills = extract_soft_skills(text)
                    resumes.append({
                        "name": name, "text": text, "skills": skills, "soft_skills": soft_skills
                    })

                matches = match_all_resumes_to_jd(jd_analysis, resumes)
                learned_scores = learned_semantic_scores(
                    jd_text, [r["text"] for r in resumes], training_model
                )

                results = []
                for resume, match, learned_score in zip(resumes, matches, learned_scores):
                    if learned_score is not None:
                        # Keep the existing keyword signal dominant while allowing
                        # historical resume language to influence semantic ranking.
                        match["learned_semantic_score"] = round(float(learned_score), 4)
                        match["final_score"] = round(100 * (
                            0.50 * match["keyword_score"] +
                            0.25 * match["semantic_score"] +
                            0.25 * float(learned_score)
                        ), 2)
                    else:
                        match["learned_semantic_score"] = None
                    results.append({"name": resume["name"], "soft_skills": resume["soft_skills"], "match": match})
                results.sort(key=lambda r: r["match"]["final_score"], reverse=True)

        st.session_state["jd_analysis"] = jd_analysis
        st.session_state["jd_bias_flags"] = jd_bias_flags
        st.session_state["results"] = results
        st.session_state["training_count"] = training_model["count"] if training_model else 0


# ----------------------------------------------------------------------------
# Recruiter chat / natural-language ranking explainer
# ----------------------------------------------------------------------------
def _candidate_from_question(question, names, excluded=None):
    """Best-effort candidate-name detection from a recruiter question."""
    excluded = excluded or set()
    q = question.lower()
    candidates = [n for n in names if n not in excluded]
    # Prefer longest names first so multi-word names win.
    for name in sorted(candidates, key=len, reverse=True):
        if name.lower() in q:
            return name
    return None


def answer_recruiter_question(question, results, jd_analysis):
    """
    Answer ranking/explanation questions from the already-computed match data.
    No LLM is used for scoring or facts: every number and skill comes from
    the matching engine's stored breakdown.
    """
    q = question.strip()
    if not q:
        return "Ask something like: “Why is Candidate A ranked above Candidate B?”"

    names = [r["name"] for r in results]
    lower = q.lower()

    # Detect two candidates for comparison questions.
    mentioned = []
    for name in sorted(names, key=len, reverse=True):
        if name.lower() in lower and name not in mentioned:
            mentioned.append(name)

    # Common single-candidate ranking questions.
    if len(mentioned) == 1:
        name = mentioned[0]
        r = next(r for r in results if r["name"] == name)
        m = r["match"]
        rank = next(i for i, x in enumerate(results, 1) if x["name"] == name)
        req_total = len(m["matched_required"]) + len(m["missing_required"])

        response = [
            f"**{name} is ranked #{rank} with a final score of {m['final_score']:.1f}/100.**",
            f"- **Required skills:** {len(m['matched_required'])}/{req_total} matched.",
            f"- **Keyword score:** {m['keyword_score']*100:.1f}%.",
            f"- **Semantic score:** {m['semantic_score']*100:.1f}%." + (f"\n- **Learned semantic score:** {m['learned_semantic_score']*100:.1f}%." if m.get("learned_semantic_score") is not None else ""),
        ]
        if m["matched_required"]:
            response.append(
                f"- **Matched required:** {format_skill_list(m['matched_required'])}."
            )
        if m["missing_required"]:
            response.append(
                f"- **Missing required:** {format_skill_list(m['missing_required'])}."
            )
        if m["matched_preferred"]:
            response.append(
                f"- **Bonus/preferred:** {format_skill_list(m['matched_preferred'])}."
            )

        if rank > 1:
            above = results[rank - 2]
            response.append(
                f"- The candidate immediately above is **{above['name']}** "
                f"at {above['match']['final_score']:.1f}/100."
            )
        return "\n".join(response)

    # Two-candidate comparison, including the requested "why X above Y" form.
    if len(mentioned) >= 2:
        a_name, b_name = mentioned[0], mentioned[1]
        a = next(r for r in results if r["name"] == a_name)
        b = next(r for r in results if r["name"] == b_name)
        ma, mb = a["match"], b["match"]

        winner, loser = (a, b) if ma["final_score"] >= mb["final_score"] else (b, a)
        mw, ml = winner["match"], loser["match"]

        req_w = len(mw["matched_required"])
        total_w = req_w + len(mw["missing_required"])
        req_l = len(ml["matched_required"])
        total_l = req_l + len(ml["missing_required"])

        lines = [
            f"**{winner['name']} ranks above {loser['name']} "
            f"({mw['final_score']:.1f} vs {ml['final_score']:.1f}).**",
            "",
            f"Here’s the breakdown:",
            f"- **Required skills:** {winner['name']} matches {req_w}/{total_w}; "
            f"{loser['name']} matches {req_l}/{total_l}.",
            f"- **Keyword score:** {mw['keyword_score']*100:.1f}% vs "
            f"{ml['keyword_score']*100:.1f}%.",
            f"- **Semantic score:** {mw['semantic_score']*100:.1f}% vs "
            f"{ml['semantic_score']*100:.1f}%.",
        ]

        only_w = mw["matched_required"] - ml["matched_required"]
        only_l = ml["matched_required"] - mw["matched_required"]
        if only_w:
            lines.append(
                f"- **Required skills {winner['name']} has that {loser['name']} "
                f"doesn't:** {format_skill_list(only_w)}."
            )
        if only_l:
            lines.append(
                f"- **Required skills {loser['name']} has that {winner['name']} "
                f"doesn't:** {format_skill_list(only_l)}."
            )
        pref_w = mw["matched_preferred"] - ml["matched_preferred"]
        if pref_w:
            lines.append(
                f"- **Additional preferred skills favoring {winner['name']}:** "
                f"{format_skill_list(pref_w)}."
            )

        score_delta = mw["final_score"] - ml["final_score"]
        if abs(score_delta) < 1:
            lines.append("- The scores are effectively tied; the ranking difference is very small.")
        elif mw["keyword_score"] > ml["keyword_score"] and mw["semantic_score"] >= ml["semantic_score"]:
            lines.append("- **Bottom line:** the higher required/preferred skill coverage, "
                         "combined with at least as strong contextual alignment, drives the ranking.")
        elif mw["semantic_score"] > ml["semantic_score"] and mw["keyword_score"] >= ml["keyword_score"]:
            lines.append("- **Bottom line:** both candidates are competitive on explicit skills, "
                         "but the higher contextual/semantic alignment gives the winner the edge.")
        else:
            lines.append("- **Bottom line:** the final score is the 50/50 weighted combination "
                         "of keyword and semantic scores, so the stronger combined result ranks higher.")

        return "\n".join(lines)

    # Generic ranking explanation.
    if any(word in lower for word in ["how", "why", "rank", "score", "ranking", "shortlist"]):
        top = results[0]
        m = top["match"]
        return (
            f"**{top['name']} is currently #1 at {m['final_score']:.1f}/100.** "
            f"The engine combines keyword and semantic scores equally. "
            f"The keyword component emphasizes required skills (80%) over preferred skills (20%), "
            f"while the semantic component uses TF-IDF context plus skill implications; when a historical resume corpus is available, a learned semantic signal is also included. "
            f"For a direct comparison, ask: **“Why is {top['name']} ranked above {names[1] if len(names) > 1 else 'Candidate Y'}?”**"
        )

    return (
        "I can explain ranking decisions from the computed match breakdown. "
        "Try: **“Why is Candidate X ranked above Candidate Y?”**, "
        "or **“Why is Candidate X ranked #2?”**"
    )


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if "results" in st.session_state:
    jd_analysis = st.session_state["jd_analysis"]
    results = st.session_state["results"]
    training_count = st.session_state.get("training_count", 0)

    if training_count > 0:
        st.markdown(
            f'<div style="color:var(--teal); font-size:0.82rem; margin-bottom:10px;">'
            f'Formula used for this run: <b>100 × (0.50 × keyword + 0.25 × semantic + '
            f'0.25 × learned)</b> — a historical corpus of {training_count} resumes was found.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:var(--ink-soft); font-size:0.82rem; margin-bottom:10px;">'
            'Formula used for this run: <b>100 × (0.5 × keyword + 0.5 × semantic)</b> — '
            'no historical resume corpus was found, so the learned signal was not applied.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">2 · Job Description Analysis</div>', unsafe_allow_html=True)
    req_pills = "".join(f'<span class="pill pill-required">{s.replace("_"," ").title()}</span>'
                         for s in sorted(jd_analysis["required_skills"]))
    pref_pills = "".join(f'<span class="pill pill-preferred">{s.replace("_"," ").title()}</span>'
                          for s in sorted(jd_analysis["preferred_skills"])) or \
                 '<span style="color:var(--ink-soft); font-size:0.85rem;">none detected</span>'
    st.markdown(f'<div style="margin-bottom:6px;"><b>Required</b></div>{req_pills}', unsafe_allow_html=True)
    st.markdown(f'<div style="margin:14px 0 6px 0;"><b>Preferred</b></div>{pref_pills}', unsafe_allow_html=True)

    bias_flags = st.session_state.get("jd_bias_flags", [])
    st.markdown('<div class="section-label">JD Bias &amp; Narrow-Phrasing Check</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:var(--ink-soft); font-size:0.85rem; margin-bottom:10px;">'
        'Heuristic, rule-based checks on the JD text itself — not a legal or HR audit, '
        'just a first pass to catch phrasing that could unfairly exclude qualified candidates.</div>',
        unsafe_allow_html=True,
    )
    if not bias_flags:
        st.markdown(
            '<div style="color:var(--ink-soft); font-size:0.9rem;">'
            'No bias or overly-narrow phrasing flags were raised for this JD.</div>',
            unsafe_allow_html=True,
        )
    else:
        for f in bias_flags:
            sev = f["severity"]
            evidence_html = (
                f'<div class="bias-evidence">Evidence: {", ".join(f["evidence"][:8])}'
                f'{"…" if len(f["evidence"]) > 8 else ""}</div>'
                if f.get("evidence") else ""
            )
            st.markdown(
                f'<div class="bias-card sev-{sev}">'
                f'<span class="bias-sev-tag sev-{sev}">{sev}</span>'
                f'<b>{f["type"].replace("_", " ").title()}</b>'
                f'<div class="bias-msg">{f["message"]}</div>'
                f'{evidence_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">3 · Ranked Shortlist</div>', unsafe_allow_html=True)
    for i, r in enumerate(results, start=1):
        m = r["match"]
        kw_pct = round(m["keyword_score"] * 100, 1)
        sem_pct = round(m["semantic_score"] * 100, 1)
        learned_pct = (
            round(m["learned_semantic_score"] * 100, 1)
            if m.get("learned_semantic_score") is not None else None
        )
        req_matched = len(m["matched_required"])
        req_total = req_matched + len(m["missing_required"])

        # Use native Streamlit layout here.  Avoid nested HTML divs because
        # Streamlit's Markdown renderer can expose closing tags as text.
        with st.container(border=True):
            rank_col, info_col, score_col = st.columns([0.55, 3.6, 0.8], gap="medium")

            with rank_col:
                rank_color = "var(--teal)" if i == 1 else "var(--ink-soft)"
                st.markdown(
                    f'<div style="font-family:Lora,serif;font-size:1.5rem;font-weight:600;color:{rank_color};padding-top:18px;">#{i}</div>',
                    unsafe_allow_html=True,
                )

            with info_col:
                st.markdown(
                    f'<div class="rank-name">{r["name"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="rank-meta">{req_matched}/{req_total} required skills matched</div>',
                    unsafe_allow_html=True,
                )

                bar_cols = st.columns(3 if learned_pct is not None else 2, gap="medium")
                bar_data = [("Keyword", kw_pct), ("Semantic", sem_pct)]
                if learned_pct is not None:
                    bar_data.append(("Learned", learned_pct))

                for bcol, (label, pct) in zip(bar_cols, bar_data):
                    with bcol:
                        st.markdown(
                            f'<div class="rank-meta">{label} {pct}%</div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(min(max(pct / 100, 0.0), 1.0), text=None)

            with score_col:
                st.markdown(
                    f'<div style="font-family:Lora,serif;font-size:1.6rem;font-weight:700;text-align:right;padding-top:14px;">{m["final_score"]:.1f}</div>',
                    unsafe_allow_html=True,
                )


    st.markdown('<div class="section-label">4 · Top 3 — Why They Ranked There</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, len(results)))
    for i, (col, r) in enumerate(zip(cols, results[:3]), start=1):
        m = r["match"]
        vclass, vlabel = verdict_class(m["required_ratio"], m["final_score"])
        matched_html = "".join(f'<span class="pill pill-required">{s.replace("_"," ").title()}</span>'
                                for s in sorted(m["matched_required"])) or "<i>none</i>"
        missing_html = "".join(f'<span class="pill" style="background:var(--coral-soft); color:var(--coral);">{s.replace("_"," ").title()}</span>'
                                for s in sorted(m["missing_required"])) or "<i>none — full coverage</i>"
        bonus_html = "".join(f'<span class="pill pill-preferred">{s.replace("_"," ").title()}</span>'
                              for s in sorted(m["matched_preferred"]))

        with col:
            # Build this card from separate Streamlit elements instead of one large
            # HTML blob.  This prevents Streamlit from displaying conditional HTML
            # as literal text in cards where optional sections differ.
            with st.container(border=True):
                head_left, head_right = st.columns([3, 1])
                with head_left:
                    st.markdown(f'<span class="explain-rank-tag">Rank #{i}</span>', unsafe_allow_html=True)
                with head_right:
                    st.markdown(
                        f'<div class="explain-score" style="text-align:right;">{m["final_score"]:.1f}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div style="font-weight:600; font-size:1.05rem; margin:6px 0 10px 0;">{r["name"]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="explain-row"><b>Matched required</b></div>', unsafe_allow_html=True)
                st.markdown(matched_html, unsafe_allow_html=True)

                st.markdown('<div class="explain-row"><b>Missing required</b></div>', unsafe_allow_html=True)
                st.markdown(missing_html, unsafe_allow_html=True)

                if bonus_html:
                    st.markdown('<div class="explain-row"><b>Bonus skills</b></div>', unsafe_allow_html=True)
                    st.markdown(bonus_html, unsafe_allow_html=True)

                # Keep the verdict as native Markdown.  Do not inject a nested HTML
                # block here; Streamlit can expose that HTML literally when cards have
                # different optional content.
                st.markdown(
                    f'**{vlabel}** · semantic standing: {round(m["semantic_score"] * 100)}% relative to this pool'
                )

    st.markdown('<div class="section-label">5 · Compare Two Candidates</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--ink-soft); font-size:0.85rem; margin-bottom:10px;">'
                'Reads directly from the match data computed above — no re-scoring.</div>', unsafe_allow_html=True)
    names = [r["name"] for r in results]
    colA, colB = st.columns(2)
    with colA:
        candidate_a = st.selectbox("Candidate A", names, index=0, label_visibility="collapsed")
    with colB:
        candidate_b = st.selectbox("Candidate B", names, index=min(1, len(names) - 1), label_visibility="collapsed")

    if candidate_a and candidate_b and candidate_a != candidate_b:
        ra = next(r for r in results if r["name"] == candidate_a)
        rb = next(r for r in results if r["name"] == candidate_b)
        ma, mb = ra["match"], rb["match"]
        winner = candidate_a if ma["final_score"] >= mb["final_score"] else candidate_b

        st.markdown(f'<div style="margin:10px 0 16px 0;"><b>{winner}</b> ranks higher '
                    f'({max(ma["final_score"], mb["final_score"]):.1f} vs '
                    f'{min(ma["final_score"], mb["final_score"]):.1f})</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        for col, name, m in [(c1, candidate_a, ma), (c2, candidate_b, mb)]:
            req_matched = len(m["matched_required"])
            req_total = req_matched + len(m["missing_required"])
            with col:
                st.markdown(textwrap.dedent(f"""
                <div class="explain-card">
                    <div style="font-weight:600; font-size:1rem; margin-bottom:8px;">{name}</div>
                    <div class="explain-row">Final score: <b>{m['final_score']:.1f}</b></div>
                    <div class="explain-row">Keyword: {round(m['keyword_score']*100,1)}% · Semantic: {round(m['semantic_score']*100,1)}%</div>
                    <div class="explain-row">Required matched: {req_matched}/{req_total}</div>
                </div>
                """), unsafe_allow_html=True)

        only_a = ma["matched_required"] - mb["matched_required"]
        only_b = mb["matched_required"] - ma["matched_required"]
        if only_a:
            st.markdown(f'<div class="explain-row"><span class="tag-ok">{candidate_a}</span> has required skills '
                        f'<span class="tag-ok">{candidate_b}</span> doesn\'t: {format_skill_list(only_a)}</div>',
                        unsafe_allow_html=True)
        if only_b:
            st.markdown(f'<div class="explain-row"><span class="tag-ok">{candidate_b}</span> has required skills '
                        f'<span class="tag-ok">{candidate_a}</span> doesn\'t: {format_skill_list(only_b)}</div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-label">6 · Ask the Recruiter Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:var(--ink-soft); font-size:0.85rem; margin-bottom:10px;">'
        'Ask why one candidate ranks above another. Answers are generated directly '
        'from the stored match breakdown — no re-scoring and no LLM judgment.</div>',
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "e.g. Why is Candidate X ranked above Candidate Y?",
        key="recruiter_chat_input",
    )
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        answer = answer_recruiter_question(prompt, results, jd_analysis)
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
        st.rerun()

else:
    st.markdown(
        '<div style="margin-top:2.5rem; color:var(--ink-soft); font-size:0.92rem;">'
        'Upload a JD and a batch of resumes above, then run matching to see the ranked shortlist.</div>',
        unsafe_allow_html=True,
    )
