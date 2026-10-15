from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_explainer import generate_evidence_note
from ranker import extract_skills, local_evidence_note, parse_skill_list, rank_resumes
from resume_parser import extract_resume_text

load_dotenv()
OPENAI_CONFIGURED = bool(os.getenv("OPENAI_API_KEY", "").strip())

st.set_page_config(page_title="Resume Match Assistant", page_icon="🧭", layout="wide")

st.title("🧭 Resume Match Assistant")
st.caption("An explainable, human-reviewed tool for comparing resume evidence with a role's stated requirements.")

with st.sidebar:
    st.header("How scoring works")
    st.markdown(
        "- **Required skills:** 65% when preferred skills are also listed; otherwise 80%.\n"
        "- **Preferred skills:** 15% when combined with required skills.\n"
        "- **Text similarity:** 20% (or adjusts when no skill list is supplied).\n\n"
        "The percentages change slightly depending on which skill lists you provide."
    )
    st.divider()
    st.subheader("Privacy by design")
    st.write(
        "Resume text is processed in the running app and is not saved to a database. "
        "Optional LLM notes receive only the role title, score components, and skill evidence—not resume text or file names."
    )

st.warning(
    "Use scores only to support a qualified human review. Do not automatically reject or select applicants from these scores. "
    "Check the original resume, validate extracted text, and apply consistent, job-related criteria to every candidate."
)

job_title = st.text_input("Role title", placeholder="e.g., Junior AI Engineer")
job_description = st.text_area(
    "Job description",
    height=190,
    placeholder="Paste the role responsibilities and qualifications here...",
    help="The description is used for local TF-IDF context similarity. Skill matching is shown separately.",
)

detected_skills = extract_skills(job_description)
if detected_skills:
    st.caption("Recognised skills in the job description: " + ", ".join(detected_skills))
else:
    st.caption("No known skills detected yet. You can still enter the role's criteria below.")

left, right = st.columns(2)
with left:
    required_raw = st.text_area(
        "Required skills (comma-separated)",
        placeholder="Python, SQL, Machine Learning",
        height=100,
        help="If you leave this blank, recognised skills from the job description are treated as required criteria.",
    )
with right:
    preferred_raw = st.text_area(
        "Preferred skills (comma-separated, optional)",
        placeholder="FastAPI, Docker, LangChain",
        height=100,
    )

uploaded_files = st.file_uploader(
    "Upload resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Text-based PDF, DOCX, and TXT files are supported. Scanned PDFs need OCR, which is not included.",
)

consent = st.checkbox(
    "I have permission to process these resumes and will use the output only as decision support.",
    value=False,
)
use_llm_notes = st.checkbox(
    "Generate short evidence notes with OpenAI (optional)",
    value=False,
    disabled=not OPENAI_CONFIGURED,
    help=(
        "Requires OPENAI_API_KEY in .env. The note generator receives structured skill evidence and scores only, not resumes or filenames."
    ),
)
if not OPENAI_CONFIGURED:
    st.info("OpenAI notes are optional. To enable them, add OPENAI_API_KEY to your .env file; local ranking works without an API key.")

if st.button("Analyze resumes", type="primary", use_container_width=False):
    st.session_state.pop("ranking_results", None)
    st.session_state.pop("processing_errors", None)

    if not job_description.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not consent:
        st.error("Please confirm you have permission to process the uploaded resumes.")
    else:
        required_skills = parse_skill_list(required_raw) if required_raw.strip() else detected_skills
        preferred_skills = parse_skill_list(preferred_raw)
        preferred_skills = [skill for skill in preferred_skills if skill not in required_skills]
        if not required_skills and not preferred_skills:
            st.warning("No recognised or manually entered skills are available. Ranking will rely only on text similarity, which may be weak.")

        candidates: list[dict[str, str]] = []
        errors: list[str] = []
        progress = st.progress(0, text="Extracting resume text...")
        for i, uploaded in enumerate(uploaded_files, start=1):
            try:
                # UploadedFile is in-memory. We don't write uploaded resumes to disk.
                text = extract_resume_text(uploaded.name, uploaded.getvalue())
                candidates.append({"filename": uploaded.name, "text": text})
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")
            progress.progress(i / len(uploaded_files), text=f"Processed {i} of {len(uploaded_files)} file(s)")
        progress.empty()

        if not candidates:
            st.error("No resumes could be read. See the file errors below.")
            st.session_state["processing_errors"] = errors
        else:
            with st.spinner("Calculating transparent match scores..."):
                results = rank_resumes(
                    job_title=job_title,
                    job_description=job_description,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills,
                    candidates=candidates,
                )

            if use_llm_notes and OPENAI_CONFIGURED:
                with st.spinner("Generating evidence notes from structured scores and skills..."):
                    for result in results:
                        try:
                            result["explanation"] = generate_evidence_note(result, job_title)
                        except Exception as exc:
                            result["explanation"] = local_evidence_note(result)
                            errors.append(
                                f"LLM note for {result['resume_file']} failed; a local note was used ({type(exc).__name__})."
                            )
            else:
                for result in results:
                    result["explanation"] = local_evidence_note(result)

            st.session_state["ranking_results"] = results
            st.session_state["processing_errors"] = errors
            st.session_state["used_required_skills"] = required_skills
            st.session_state["used_preferred_skills"] = preferred_skills

if st.session_state.get("processing_errors"):
    with st.expander("File or optional LLM processing notes", expanded=True):
        for error in st.session_state["processing_errors"]:
            st.write("- " + error)

results = st.session_state.get("ranking_results")
if results:
    st.divider()
    st.header("Results")
    st.write(
        "Higher scores indicate closer matches to the criteria entered. They are not probabilities of success and should not be compared across unrelated job descriptions."
    )
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Resumes evaluated", len(results))
    metric2.metric("Top advisory score", f"{results[0]['overall_score']:.2f}%")
    metric3.metric("Required skills used", len(st.session_state.get("used_required_skills", [])))

    display_rows = []
    for result in results:
        display_rows.append({
            "Rank": result["rank"],
            "Resume file": result["resume_file"],
            "Overall match (%)": result["overall_score"],
            "Required coverage (%)": result["required_coverage"],
            "Preferred coverage (%)": result["preferred_coverage"],
            "Text similarity (%)": result["text_similarity"],
            "Required skills found": ", ".join(result["matched_required"]),
            "Required skills not found in text": ", ".join(result["missing_required"]),
            "Preferred skills found": ", ".join(result["matched_preferred"]),
        })
    result_df = pd.DataFrame(display_rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    csv_df = pd.DataFrame([
        {
            "rank": r["rank"],
            "resume_file": r["resume_file"],
            "overall_match_percent": r["overall_score"],
            "required_coverage_percent": r["required_coverage"],
            "preferred_coverage_percent": r["preferred_coverage"],
            "text_similarity_percent": r["text_similarity"],
            "matched_required_skills": "; ".join(r["matched_required"]),
            "required_skills_not_found_in_text": "; ".join(r["missing_required"]),
            "matched_preferred_skills": "; ".join(r["matched_preferred"]),
            "preferred_skills_not_found_in_text": "; ".join(r["missing_preferred"]),
            "evidence_note": r["explanation"],
        }
        for r in results
    ])
    st.download_button(
        "Download results as CSV",
        data=csv_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resume_match_results.csv",
        mime="text/csv",
    )

    st.subheader("Candidate-by-candidate evidence")
    for result in results:
        with st.expander(f"#{result['rank']} · {result['resume_file']} · {result['overall_score']:.2f}% advisory match"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Required skills coverage", "—" if result["required_coverage"] is None else f"{result['required_coverage']:.2f}%")
            c2.metric("Preferred skills coverage", "—" if result["preferred_coverage"] is None else f"{result['preferred_coverage']:.2f}%")
            c3.metric("Text similarity", f"{result['text_similarity']:.2f}%")
            st.markdown("**Required skills found in extracted text**")
            st.write(", ".join(result["matched_required"]) if result["matched_required"] else "None found")
            st.markdown("**Required skills not found in extracted text**")
            st.write(", ".join(result["missing_required"]) if result["missing_required"] else "None")
            st.markdown("**Preferred skills found in extracted text**")
            st.write(", ".join(result["matched_preferred"]) if result["matched_preferred"] else "None found")
            st.markdown("**Evidence note**")
            st.write(result["explanation"])

    st.caption("A skill marked 'not found' may simply be absent from extracted text. Always inspect the original resume before drawing conclusions.")
