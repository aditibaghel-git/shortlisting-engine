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

The ranking combines:

- Keyword matching for explicit skills and requirements
- Semantic similarity using TF-IDF and cosine similarity
- Fuzzy matching and skill relationships to handle variations

## Project Structure

```text
project/
├── src/
│   ├── app.py
│   ├── parser.py
│   ├── skill_extractor.py
│   ├── skill_taxonomy.py
│   ├── jd_analyzer.py
│   ├── matcher.py
│   ├── explainer.py
│   ├── bias_detector.py
│   └── pipeline.py
├── data/
│   ├── jds/
│   └── resumes/
└── output/
```

## Installation

```bash
pip install streamlit pandas scikit-learn rapidfuzz pypdf
```

## Run

```bash
cd src
streamlit run app.py
```

## Command Line

```bash
cd src
python pipeline.py <path_to_jd> <path_to_resume_folder>
```

## Recruiter Chat

The application includes a chat interface for questions about the shortlist.

Examples:

- Why is this candidate ranked higher?
- What required skills are missing?
- What are this candidate's strongest matches?
- Which candidate has better skill coverage?

## Fairness Checks

Job descriptions can be checked for potentially exclusionary or overly narrow wording, including:

- Age-related wording
- Gendered wording
- National-origin-related wording
- Subjective culture or prestige language

These are review flags, not legal conclusions.

## Resume Robustness

The parser and skill extraction components handle common resume inconsistencies such as:

- Different section headings
- Minor spelling variations
- Different skill formats
- Inconsistent formatting
- Common date and experience formats

## Design Principles

- **Explainable** — results are easy to understand
- **Deterministic** — consistent results for the same inputs
- **Modular** — components are separated by responsibility
- **Practical** — focused on information available in resumes and job descriptions

## Limitations

- Fairness checks are heuristic and require human review.
- The system is intended to support, not replace, human decision-making.

## Future Improvements

- Stronger semantic models
- Better experience and date extraction
- Expanded skill taxonomy
- Improved fairness detection
- More advanced candidate evaluation
