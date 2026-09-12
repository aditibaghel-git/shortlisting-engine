"""
End-to-end pipeline:
  1. Load JD, analyze into required/preferred skills
  2. Load all resumes, extract text + skills
  3. Match each resume against JD (keyword + semantic)
  4. Rank all candidates
  5. Generate explanations for top 3
  6. Output as JSON + printed report
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(__file__))

from parser import extract_text, extract_candidate_name
from jd_analyzer import analyze_jd
from skill_extractor import extract_skills, extract_soft_skills
from matcher import match_all_resumes_to_jd
from explainer import generate_explanation


def load_jd(jd_path: str) -> dict:
    jd_text = extract_text(jd_path)
    return analyze_jd(jd_text)


def load_resumes(resume_dir: str) -> list:
    """Load all PDF/txt resumes from a directory."""
    paths = sorted(glob.glob(os.path.join(resume_dir, "*.pdf")) +
                    glob.glob(os.path.join(resume_dir, "*.txt")))
    resumes = []
    for path in paths:
        text = extract_text(path)
        name = extract_candidate_name(text, fallback=os.path.splitext(os.path.basename(path))[0])
        skills = extract_skills(text)
        soft_skills = extract_soft_skills(text)
        resumes.append({
            "path": path,
            "name": name,
            "text": text,
            "skills": skills,
            "soft_skills": soft_skills,
        })
    return resumes


def run_pipeline(jd_path: str, resume_dir: str, top_n_explanations: int = 3) -> dict:
    jd_analysis = load_jd(jd_path)
    resumes = load_resumes(resume_dir)

    if not resumes:
        raise ValueError(f"No resumes found in {resume_dir}")

    matches = match_all_resumes_to_jd(jd_analysis, resumes)
    results = []
    for resume, match in zip(resumes, matches):
        results.append({
            "name": resume["name"],
            "path": resume["path"],
            "soft_skills": resume["soft_skills"],
            "match": match,
        })

    # Rank descending by final_score
    results.sort(key=lambda r: r["match"]["final_score"], reverse=True)

    # Generate explanations for top N
    explanations = []
    for i, r in enumerate(results[:top_n_explanations], start=1):
        explanation_text = generate_explanation(r["name"], i, r["match"])
        explanations.append({
            "rank": i,
            "name": r["name"],
            "explanation": explanation_text,
        })

    ranking_table = [
        {
            "rank": i + 1,
            "name": r["name"],
            "final_score": r["match"]["final_score"],
            "keyword_score": round(r["match"]["keyword_score"] * 100, 1),
            "semantic_score": round(r["match"]["semantic_score"] * 100, 1),
            "required_skills_matched": f"{len(r['match']['matched_required'])}/"
                                        f"{len(r['match']['matched_required']) + len(r['match']['missing_required'])}",
        }
        for i, r in enumerate(results)
    ]

    return {
        "jd_required_skills": sorted(jd_analysis["required_skills"]),
        "jd_preferred_skills": sorted(jd_analysis["preferred_skills"]),
        "ranking_table": ranking_table,
        "top_explanations": explanations,
        "raw_results": results,
    }


def print_report(output: dict):
    print("=" * 70)
    print("JOB DESCRIPTION ANALYSIS")
    print("=" * 70)
    print(f"Required skills: {', '.join(output['jd_required_skills'])}")
    print(f"Preferred skills: {', '.join(output['jd_preferred_skills']) or 'none detected'}")

    print("\n" + "=" * 70)
    print("RANKED SHORTLIST")
    print("=" * 70)
    print(f"{'Rank':<6}{'Name':<25}{'Final':<8}{'Keyword':<10}{'Semantic':<10}{'Req.Match':<10}")
    for row in output["ranking_table"]:
        print(f"{row['rank']:<6}{row['name']:<25}{row['final_score']:<8}"
              f"{row['keyword_score']:<10}{row['semantic_score']:<10}{row['required_skills_matched']:<10}")

    print("\n" + "=" * 70)
    print("TOP 3 EXPLANATIONS")
    print("=" * 70)
    for exp in output["top_explanations"]:
        print(exp["explanation"])
        print("-" * 70)


if __name__ == "__main__":
    JD_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/jds/sample_jd.txt"
    RESUME_DIR = sys.argv[2] if len(sys.argv) > 2 else "../data/resumes"

    output = run_pipeline(JD_PATH, RESUME_DIR)
    print_report(output)

    # Save JSON output (without raw resume text to keep it clean)
    json_output = {
        "jd_required_skills": output["jd_required_skills"],
        "jd_preferred_skills": output["jd_preferred_skills"],
        "ranking_table": output["ranking_table"],
        "top_explanations": output["top_explanations"],
    }
    out_path = "../output/shortlist_result.json"
    with open(out_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nSaved JSON output to {out_path}")
