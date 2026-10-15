"""Optional OpenAI explanations based on already-computed, non-identifying evidence.

The function intentionally does not send resume contents, file names, contact
information, or candidate identity to the language model.
"""

from __future__ import annotations

import os


def generate_evidence_note(result: dict[str, object], job_title: str = "") -> str:
    """Use the OpenAI Responses API to explain the structured score evidence."""
    from openai import OpenAI  # optional import; only needed when user enables it

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    # Only structured skill evidence and numeric component scores are sent.
    payload = {
        "role_title": job_title,
        "overall_match_score_percent": result.get("overall_score"),
        "required_skill_coverage_percent": result.get("required_coverage"),
        "preferred_skill_coverage_percent": result.get("preferred_coverage"),
        "text_similarity_percent": result.get("text_similarity"),
        "required_skills_found_in_text": result.get("matched_required", []),
        "required_skills_not_found_in_text": result.get("missing_required", []),
        "preferred_skills_found_in_text": result.get("matched_preferred", []),
        "preferred_skills_not_found_in_text": result.get("missing_preferred", []),
    }

    instructions = (
        "You write short, neutral HR review notes based only on structured, job-related evidence. "
        "Do not infer personality, age, gender, race, ethnicity, disability, religion, family status, "
        "nationality, or any other protected/personal characteristic. Do not make a hire/reject decision. "
        "Do not claim a skill is absent from the person's real abilities; say only that it was not found "
        "in the extracted resume text. Mention both supporting evidence and gaps. If the evidence is sparse, "
        "say that a human should inspect the original resume. Keep it to 2-4 sentences."
    )
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input="Create a factual, non-final review note from this structured evidence:\n" + str(payload),
        max_output_tokens=180,
        store=False,
    )
    note = (response.output_text or "").strip()
    if not note:
        raise RuntimeError("The LLM returned an empty explanation.")
    return note
