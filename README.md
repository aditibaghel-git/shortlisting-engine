# Smart Shortlisting Engine

An explainable resume screening and candidate-ranking tool that matches resumes against a job description.

## Features

- **Hybrid candidate ranking** using keyword and semantic similarity
- **Required vs. preferred skill matching**
- **Explainable results** showing why candidates rank where they do
- **Recruiter chat** for questions about rankings and candidate matches
- **Messy resume handling** for inconsistent formatting and skill variations
- **Job description fairness checks** for potentially narrow wording

## How It Works

1. Upload a job description.
2. Upload one or more resumes.
3. Extract skills, experience, and relevant text.
4. Compare each resume with the job description.
5. Calculate a match score.
6. Rank candidates and provide explanations.

### Matching

The ranking combines two independently-computed scores:

**Keyword score** — explicit skill/tool overlap:

```
keyword_score = 0.8 × required_ratio + 0.2 × preferred_ratio
```

where `required_ratio` and `preferred_ratio` are the fraction of the JD's
required / preferred skills found in the resume, via exact alias matching
against the skill taxonomy plus fuzzy matching (RapidFuzz) for typos and
minor variants.

**Semantic score** — contextual/meaning overlap, not just literal words:

```
semantic_score = 0.75 × tfidf_similarity + 0.25 × implication_boost
```

`tfidf_similarity` comes from TF-IDF (unigrams + bigrams) cosine
similarity, fit jointly across the JD and the *entire* resume pool and
then min-max normalized across that pool — this is what lets "built REST
APIs with Express and MongoDB" score well against a Node.js JD even
without the literal word "Node." `implication_boost` rewards resumes
whose skills imply an unmatched required skill (e.g. Express implies
Node.js/REST competency).

**Final score:**

```
final_score = 100 × (0.5 × keyword_score + 0.5 × semantic_score)
```

The Streamlit app can optionally blend in a third, learned TF-IDF signal
fit on a historical resume corpus when one is available, using
`0.50 × keyword + 0.25 × semantic + 0.25 × learned`; it falls back to the
50/50 formula above when no historical corpus is present. Weights are
tunable constants (`KEYWORD_WEIGHT`, `SEMANTIC_WEIGHT`, etc.) documented
in `matcher.py`, not a fixed/hardcoded design.

## Project Structure

```text
project/
├── app.py
├── parser.py
├── skill_extractor.py
├── skill_taxonomy.py
├── jd_analyzer.py
├── matcher.py
├── explainer.py
├── bias_flagger.py
└── pipeline.py
```

## Installation

```bash
pip install streamlit scikit-learn rapidfuzz pdfplumber numpy
```

## Run

```bash
streamlit run app.py
```

## Command Line

```bash
python pipeline.py <path_to_jd> <path_to_resume_folder>
```

## Recruiter Chat

The application includes a chat interface for questions about the shortlist.
Answers are generated directly from the stored match breakdown — no
re-scoring and no LLM judgment.

Examples:

- Why is this candidate ranked higher?
- What required skills are missing?
- What are this candidate's strongest matches?
- Which candidate has better skill coverage?

## Fairness Checks

Job descriptions are checked for potentially exclusionary or overly narrow
wording, including:

- Age-coded wording (e.g. "digital native," "young and energetic," a bare "young" describing a desired trait)
- Gendered wording (masculine- or feminine-coded language skew)
- National-origin / language wording (e.g. "native English speaker," "no accent")
- Affinity bias (vague "culture fit" language, and elite-pedigree/prestige phrasing like "top university")
- Seniority/experience mismatches (e.g. an "intern" role demanding 3+ years)
- Degree requirements with no "or equivalent experience" allowance
- Kitchen-sink required-skill lists that may discourage qualified applicants
- Single-vendor cloud lock-in (e.g. requiring AWS specifically over "cloud platform experience")

Each flag is produced by a concrete, inspectable rule — a keyword list,
regex, or threshold — not an LLM judgment, and comes with a severity
(high/medium/low) and supporting evidence. These are review flags, not
legal conclusions.

## Resume Robustness

The parser and skill extraction components handle common resume
inconsistencies such as:

- Different section headings (handled via separator-agnostic header
  matching — "MUST-HAVE," "must_have," and "must have" all resolve the same way)
- Minor spelling variations (fuzzy matching)
- Different skill formats and aliases (skill taxonomy)
- Inconsistent formatting, hyphenated line-breaks, and excess whitespace
- Common "N years of experience" phrasing

## Design Principles

- **Explainable** — results are easy to understand
- **Deterministic** — consistent results for the same inputs
- **Modular** — components are separated by responsibility
- **Practical** — focused on information available in resumes and job descriptions

## Limitations

- Fairness checks are heuristic and require human review.
- TF-IDF semantic scores are relative to the current resume pool — the
  same resume can score differently against a different batch, which is
  expected for a ranking task but worth knowing when comparing across runs.
- The system is intended to support, not replace, human decision-making.

## Future Improvements

- Stronger semantic models
- Better experience and date extraction
- Expanded skill taxonomy
- Improved fairness detection
- More advanced candidate evaluation
