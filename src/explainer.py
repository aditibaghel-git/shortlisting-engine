"""
Generate short, human-readable explanations for why a candidate ranked where they did.
Pure rule-based generation from the match breakdown.

"""


def format_skill_list(skills: set) -> str:
    if not skills:
        return "none"
    return ", ".join(sorted(s.replace("_", " ").title() for s in skills))


def generate_explanation(candidate_name: str, rank: int, match_result: dict) -> str:
    matched_req = match_result["matched_required"]
    missing_req = match_result["missing_required"]
    matched_pref = match_result["matched_preferred"]
    score = match_result["final_score"]
    req_ratio = match_result["required_ratio"]
    tfidf = match_result["tfidf_similarity"]

    lines = []
    lines.append(f"Rank #{rank}: {candidate_name} — Final Score: {score:.1f}/100")

    if matched_req:
        lines.append(f"✅ Matched required skills: {format_skill_list(matched_req)} "
                      f"({len(matched_req)}/{len(matched_req) + len(missing_req)} required skills covered).")
    else:
        lines.append("⚠️ No required skills were directly matched by keyword.")

    if missing_req:
        lines.append(f"❌ Missing required skills: {format_skill_list(missing_req)}.")
    else:
        lines.append("✅ All required skills are covered.")

    if matched_pref:
        lines.append(f"➕ Also brings preferred/bonus skills: {format_skill_list(matched_pref)}.")

    # Contextual/semantic note. Note: tfidf here is normalized RELATIVE to the
    # candidate pool (see matcher.compute_semantic_similarities_batch), so these
    # thresholds describe standing within this batch, not an absolute score.
    if tfidf >= 0.66:
        lines.append("📄 Resume content also shows strong contextual alignment with the JD's "
                      "language and focus areas, relative to the rest of this candidate pool.")
    elif tfidf >= 0.33:
        lines.append("📄 Resume shows moderate contextual alignment with the JD relative to "
                      "the rest of this candidate pool.")
    else:
        lines.append("📄 Resume shows relatively low contextual overlap with the JD "
                      "compared to other candidates in this pool.")

    if req_ratio == 1.0 and score >= 70:
        lines.append("Overall: strong fit — meets all required skills with good contextual alignment.")
    elif req_ratio >= 0.5:
        lines.append("Overall: partial fit — covers a majority of required skills but has some gaps.")
    else:
        lines.append("Overall: weak fit — significant gaps in required skills for this role.")

    return "\n".join(lines)
