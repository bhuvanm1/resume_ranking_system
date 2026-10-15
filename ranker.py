"""Transparent, job-related resume matching utilities.

This module deliberately avoids making hiring decisions. It produces an advisory
match score from explicitly listed skills and TF-IDF text similarity so a human
reviewer can inspect the underlying evidence.
"""

from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Canonical skill name -> aliases that may appear in a resume or job description.
# This is intentionally a small, editable dictionary, not a universal skills ontology.
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "SQL": ("sql",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "MongoDB": ("mongodb", "mongo db"),
    "Redis": ("redis",),
    "Pandas": ("pandas",),
    "NumPy": ("numpy", "numpy library"),
    "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "TensorFlow": ("tensorflow", "tf.keras"),
    "PyTorch": ("pytorch", "torch"),
    "Keras": ("keras",),
    "Machine Learning": ("machine learning", "ml models"),
    "Deep Learning": ("deep learning",),
    "Natural Language Processing": ("natural language processing", "nlp"),
    "LLM": ("llm", "llms", "large language model", "large language models"),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "Prompt Engineering": ("prompt engineering", "prompt design"),
    "LangChain": ("langchain",),
    "LangGraph": ("langgraph",),
    "Hugging Face": ("hugging face", "huggingface"),
    "Transformers": ("transformers", "transformer architecture"),
    "FAISS": ("faiss",),
    "Chroma": ("chroma", "chromadb"),
    "Pinecone": ("pinecone",),
    "Vector Database": ("vector database", "vector databases", "vector store", "vector stores"),
    "OpenAI API": ("openai api", "openai sdk"),
    "REST API": ("rest api", "rest apis", "restful api", "restful apis"),
    "FastAPI": ("fastapi", "fast api"),
    "Flask": ("flask",),
    "Django": ("django",),
    "Streamlit": ("streamlit",),
    "OpenCV": ("opencv", "cv2"),
    "CNN": ("cnn", "convolutional neural network", "convolutional neural networks"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "AWS": ("aws", "amazon web services"),
    "Microsoft Azure": ("microsoft azure", "azure"),
    "Google Cloud": ("google cloud", "gcp"),
    "Git": ("git",),
    "GitHub": ("github",),
    "Linux": ("linux",),
    "CI/CD": ("ci/cd", "continuous integration", "continuous delivery", "continuous deployment"),
    "Java": ("java",),
    "JavaScript": ("javascript", "java script"),
    "TypeScript": ("typescript",),
    "React": ("react", "react.js", "reactjs"),
    "Node.js": ("node.js", "nodejs", "node js"),
    "C++": ("c++", "cpp"),
    "C#": ("c#", "c sharp"),
    ".NET": (".net", "dotnet", "asp.net"),
    "HTML": ("html",),
    "CSS": ("css",),
    "GraphQL": ("graphql",),
    "Apache Spark": ("apache spark", "pyspark", "spark"),
    "Hadoop": ("hadoop",),
    "Tableau": ("tableau",),
    "Power BI": ("power bi", "powerbi"),
    "Microsoft Excel": ("microsoft excel", "excel", "excel pivot tables"),
    "Data Analysis": ("data analysis", "data analytics"),
    "Data Visualization": ("data visualization", "data visualisation"),
    "Statistics": ("statistics", "statistical analysis"),
    "ETL": ("etl", "extract transform load"),
    "Apache Airflow": ("apache airflow", "airflow"),
    "MLflow": ("mlflow",),
    "Model Evaluation": ("model evaluation", "model validation"),
    "A/B Testing": ("a/b testing", "ab testing", "split testing"),
    "Data Structures and Algorithms": ("data structures and algorithms", "data structures", "algorithms", "dsa"),
}


def _pattern_for_phrase(phrase: str) -> re.Pattern[str]:
    """Compile a case-insensitive phrase matcher that tolerates repeated spaces."""
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    # Boundaries avoid finding e.g. Java inside JavaScript, while punctuation in
    # names such as C++, C#, and Node.js is still allowed.
    return re.compile(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE)


_COMPILED_ALIASES: dict[str, tuple[re.Pattern[str], ...]] = {
    canonical: tuple(_pattern_for_phrase(alias) for alias in aliases)
    for canonical, aliases in SKILL_ALIASES.items()
}


def extract_skills(text: str) -> list[str]:
    """Return known skills mentioned in text, in the dictionary's stable order."""
    if not text:
        return []
    found: list[str] = []
    for canonical, patterns in _COMPILED_ALIASES.items():
        if any(pattern.search(text) for pattern in patterns):
            found.append(canonical)
    return found


def _normalise_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


_ALIAS_TO_CANONICAL = {
    _normalise_label(alias): canonical
    for canonical, aliases in SKILL_ALIASES.items()
    for alias in (canonical, *aliases)
}


def parse_skill_list(raw: str | Iterable[str] | None) -> list[str]:
    """Parse comma/newline-separated skills and canonicalise known aliases.

    User-entered skills that are not in SKILL_ALIASES are preserved as custom
    criteria and matched as literal phrases.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[,\n;]", raw)
    else:
        items = list(raw)

    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = str(item).strip()
        if not label:
            continue
        canonical = _ALIAS_TO_CANONICAL.get(_normalise_label(label), label)
        key = _normalise_label(canonical)
        if key not in seen:
            seen.add(key)
            output.append(canonical)
    return output


def _skill_pattern(skill: str) -> re.Pattern[str]:
    known = _COMPILED_ALIASES.get(skill)
    if known:
        # A canonical skill can have several common spellings.
        aliases = SKILL_ALIASES[skill]
        alternatives = "|".join(
            re.escape(alias).replace(r"\ ", r"\s+") for alias in sorted(aliases, key=len, reverse=True)
        )
        return re.compile(r"(?<![A-Za-z0-9])(?:" + alternatives + r")(?![A-Za-z0-9])", re.IGNORECASE)
    return _pattern_for_phrase(skill)


def find_skill_evidence(text: str, skills: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split requested skills into present and absent evidence sets."""
    matched: list[str] = []
    missing: list[str] = []
    for skill in skills:
        if _skill_pattern(skill).search(text or ""):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def _clean_text(text: str) -> str:
    """Remove common contact fields before text similarity is calculated."""
    text = text or ""
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:\+?\d[\d\s().-]{7,}\d)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _text_similarities(job_text: str, resume_texts: list[str]) -> list[float]:
    """Calculate local TF-IDF cosine similarities between the role and resumes."""
    if not resume_texts:
        return []
    cleaned_job = _clean_text(job_text)
    cleaned_resumes = [_clean_text(text) for text in resume_texts]
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([cleaned_job, *cleaned_resumes])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return [float(max(0.0, min(1.0, score))) for score in similarities]
    except ValueError:
        # E.g. an empty job description after tokenisation.
        return [0.0] * len(resume_texts)


def rank_resumes(
    job_title: str,
    job_description: str,
    required_skills: Iterable[str],
    preferred_skills: Iterable[str],
    candidates: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Rank candidate text against role criteria using explicit, explainable rules.

    Weights:
      * Required skills + preferred skills present: 65% required, 15% preferred,
        20% TF-IDF context similarity.
      * Required skills only: 80% required, 20% TF-IDF context similarity.
      * Preferred skills only: 60% preferred, 40% TF-IDF context similarity.
      * No skill criteria: 100% TF-IDF context similarity.

    Scores are relative screening aids, not probabilities or hiring decisions.
    """
    required = parse_skill_list(required_skills)
    preferred = [skill for skill in parse_skill_list(preferred_skills) if skill not in required]
    role_text = "\n".join([job_title or "", job_description or "", *required, *preferred])
    similarities = _text_similarities(role_text, [candidate.get("text", "") for candidate in candidates])

    results: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        candidate_text = candidate.get("text", "")
        matched_required, missing_required = find_skill_evidence(candidate_text, required)
        matched_preferred, missing_preferred = find_skill_evidence(candidate_text, preferred)
        required_coverage = len(matched_required) / len(required) if required else None
        preferred_coverage = len(matched_preferred) / len(preferred) if preferred else None
        similarity = similarities[index] if index < len(similarities) else 0.0

        if required and preferred:
            score = 0.65 * required_coverage + 0.15 * preferred_coverage + 0.20 * similarity
        elif required:
            score = 0.80 * required_coverage + 0.20 * similarity
        elif preferred:
            score = 0.60 * preferred_coverage + 0.40 * similarity
        else:
            score = similarity

        results.append({
            "resume_file": candidate.get("filename", "uploaded_resume"),
            "overall_score": round(float(score) * 100, 2),
            "required_coverage": round(required_coverage * 100, 2) if required_coverage is not None else None,
            "preferred_coverage": round(preferred_coverage * 100, 2) if preferred_coverage is not None else None,
            "text_similarity": round(similarity * 100, 2),
            "matched_required": matched_required,
            "missing_required": missing_required,
            "matched_preferred": matched_preferred,
            "missing_preferred": missing_preferred,
            "explanation": "",
        })

    results.sort(
        key=lambda row: (
            float(row["overall_score"]),
            float(row["required_coverage"] or 0),
            str(row["resume_file"]).casefold(),
        ),
        reverse=True,
    )
    for position, row in enumerate(results, start=1):
        row["rank"] = position
    return results


def local_evidence_note(result: dict[str, object]) -> str:
    """Produce a deterministic evidence note without any LLM/API calls."""
    matched = result.get("matched_required", [])
    missing = result.get("missing_required", [])
    preferred = result.get("matched_preferred", [])
    parts: list[str] = []
    if matched:
        parts.append("Required skill evidence found: " + ", ".join(matched) + ".")
    if missing:
        parts.append("Required skills not found in extracted text: " + ", ".join(missing) + ".")
    if preferred:
        parts.append("Preferred skill evidence found: " + ", ".join(preferred) + ".")
    if not parts:
        parts.append("No explicit skills were matched; review the original resume and job-related evidence manually.")
    parts.append("This is a screening aid, not a recommendation to hire or reject.")
    return " ".join(parts)
