"""Agent 4: turn the structured eligibility assessments into a plain-language answer."""

from src.agents.state import NavigatorState
from src.llm import get_llm
from src.schemas import EligibilityAssessment, UserProfile

SYSTEM_PROMPT = """You are the explainer agent for a EU social-benefits navigator.
You will be given a user's profile and a list of eligibility assessments for different
benefit schemes (each with a verdict and justification already reasoned out by another agent).

Write a warm, plain-language answer for the user, in the SAME LANGUAGE they wrote in
(language code will be given). Structure it as:
1. A short empathetic opening sentence.
2. For each scheme that is "Eligible" or "Possibly Eligible": the scheme name, country,
   a one-line plain-language reason, and a concrete next step (where/how to apply).
3. For schemes marked "Insufficient Info": say what extra detail would help.
4. Skip schemes marked "Not Eligible" unless nothing else qualified, in which case briefly
   explain why and suggest they double-check locally as rules change.
5. End with one sentence reminding them this is general guidance, not legal advice, and
   rules/thresholds change — they should confirm on the official page before applying.

Keep it concise and skimmable. Do not invent facts beyond what's in the assessments.
"""


def _format_assessments(assessments: list[EligibilityAssessment]) -> str:
    parts = []
    for a in assessments:
        parts.append(
            f"- Scheme: {a.scheme_name} ({a.country})\n"
            f"  Verdict: {a.verdict.value}\n"
            f"  Justification: {a.justification}\n"
            f"  Sources: {', '.join(a.citations)}"
        )
    return "\n".join(parts) if parts else "No candidate schemes were found."


def run_explainer(profile: UserProfile, assessments: list[EligibilityAssessment]) -> str:
    llm = get_llm()
    user_content = (
        f"User's language code: {profile.language}\n"
        f"User's situation: {profile.raw_situation}\n\n"
        f"Eligibility assessments:\n{_format_assessments(assessments)}"
    )
    return llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        role="fast",
        temperature=0.4,
    )


def explainer_node(state: NavigatorState) -> dict:
    draft = run_explainer(state["profile"], state.get("assessments", []))
    trace = state.get("trace", []) + [
        "**Explainer Agent** — drafted a plain-language answer with next steps, "
        f"in language '{state['profile'].language}'"
    ]
    return {"draft_answer": draft, "trace": trace}
