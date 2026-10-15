from ranker import extract_skills, find_skill_evidence, parse_skill_list, rank_resumes


def test_extract_skills_and_aliases():
    text = "Built a RAG application with Python, sklearn, FastAPI and a Chroma vector store."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "scikit-learn" in skills
    assert "RAG" in skills
    assert "FastAPI" in skills
    assert "Chroma" in skills


def test_parse_skill_list_deduplicates_and_canonicalises():
    parsed = parse_skill_list("python, Python, sklearn\nSQL")
    assert parsed == ["Python", "scikit-learn", "SQL"]


def test_find_skill_evidence():
    matched, missing = find_skill_evidence("I use Python and FastAPI", ["Python", "SQL", "FastAPI"])
    assert matched == ["Python", "FastAPI"]
    assert missing == ["SQL"]


def test_rank_resumes_returns_sorted_scores_and_evidence():
    candidates = [
        {"filename": "candidate_b.txt", "text": "Python, Pandas, NumPy, scikit-learn, machine learning, REST API."},
        {"filename": "candidate_a.txt", "text": "Power BI, Excel, Tableau, SQL."},
    ]
    results = rank_resumes(
        "Junior ML Engineer",
        "Build models with Python and scikit-learn and expose them with a REST API.",
        ["Python", "scikit-learn", "REST API"],
        ["Pandas", "NumPy"],
        candidates,
    )
    assert len(results) == 2
    assert results[0]["resume_file"] == "candidate_b.txt"
    assert results[0]["overall_score"] >= results[1]["overall_score"]
    assert "Python" in results[0]["matched_required"]
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
