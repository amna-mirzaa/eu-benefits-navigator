"""Agent 3: for each candidate scheme, reason step-by-step over eligibility.

Uses the Groq `reasoning` model (openai/gpt-oss-120b, with reasoning_effort="high")
so the chain-of-thought is a real model capability, not just a prompted persona.
"""

import re

from src.agents.state import NavigatorState
from src.llm import get_llm
from src.schemas import EligibilityAssessment, EligibilityVerdict, SourceChunk, UserProfile

SYSTEM_PROMPT = """You are the eligibility-reasoning agent for a EU social-benefits navigator.
You will be given a user's profile and the retrieved official rule text for ONE benefit scheme.

Reason step by step: go through each eligibility criterion present in the rule text and
explicitly compare it against the user's stated facts. If a fact needed to decide a
criterion is missing from the profile, say so explicitly — do not guess or assume.
Do not invent eligibility rules that are not present in the provided rule text.

After your reasoning, output one final line, starting exactly with `FINAL_JSON:`,
followed by a single-line JSON object with exactly these keys:
{"verdict": "Eligible" | "Possibly Eligible" | "Not Eligible" | "Insufficient Info", "justification": "one or two plain-language sentences"}
"""


# Tolerant of markdown formatting some models wrap the marker in (e.g.
# "**FINAL_JSON:**") and assumes a flat (non-nested) JSON object, so the
# lazy `.*?` only has to skip past stray markdown characters, not braces.
FINAL_JSON_RE = re.compile(r"FINAL_JSON:.*?(\{[^{}]*\})", re.DOTALL)


def _group_by_scheme(chunks: list[SourceChunk]) -> dict[tuple[str, str, str], list[SourceChunk]]:
    groups: dict[tuple[str, str, str], list[SourceChunk]] = {}
    for chunk in chunks:
        key = (chunk.country, chunk.scheme_name, chunk.category)
        groups.setdefault(key, []).append(chunk)
    return groups


def _profile_summary(profile: UserProfile) -> str:
    return (
        f"Country: {profile.country or 'unknown'}\n"
        f"Age: {profile.age or 'unknown'}\n"
        f"Employment status: {profile.employment_status or 'unknown'}\n"
        f"Household: {profile.household or 'unknown'}\n"
        f"Income band: {profile.income_band or 'unknown'}\n"
        f"Disability: {profile.disability if profile.disability is not None else 'unknown'}\n"
        f"Residency status: {profile.residency_status or 'unknown'}\n"
        f"Additional notes: {profile.notes or 'none'}\n"
        f"Original description: {profile.raw_situation}"
    )


def assess_scheme(
    profile: UserProfile, country: str, scheme_name: str, category: str, chunks: list[SourceChunk]
) -> EligibilityAssessment:
    rule_text = "\n\n".join(c.text for c in chunks)
    user_content = (
        f"USER PROFILE:\n{_profile_summary(profile)}\n\n"
        f"BENEFIT SCHEME: {scheme_name} ({country})\n"
        f"RETRIEVED RULE TEXT:\n{rule_text}"
    )

    llm = get_llm()
    content, reasoning = llm.complete_with_reasoning(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )

    match = FINAL_JSON_RE.search(content)
    verdict = EligibilityVerdict.INSUFFICIENT_INFO
    justification = content.strip()[-400:]
    reasoning_trace = reasoning.strip() if reasoning else content.strip()

    if match:
        import json

        try:
            parsed = json.loads(match.group(1))
            verdict = EligibilityVerdict(parsed.get("verdict", verdict.value))
            justification = parsed.get("justification", justification)
        except (json.JSONDecodeError, ValueError):
            pass
        if not reasoning:
            reasoning_trace = content[: match.start()].strip()

    citations = sorted({c.source_url for c in chunks})
    return EligibilityAssessment(
        scheme_name=scheme_name,
        country=country,
        category=category,
        verdict=verdict,
        reasoning_trace=reasoning_trace,
        justification=justification,
        citations=citations,
        source_chunks=chunks[:3],
    )


def eligibility_node(state: NavigatorState) -> dict:
    profile = state["profile"]
    chunks = state.get("retrieved_chunks", [])
    groups = _group_by_scheme(chunks)

    assessments = [
        assess_scheme(profile, country, scheme_name, category, scheme_chunks)
        for (country, scheme_name, category), scheme_chunks in groups.items()
    ]

    trace = state.get("trace", []) + [
        f"**Eligibility Reasoning Agent** — assessed {len(assessments)} candidate scheme(s) "
        f"using step-by-step reasoning: "
        + ", ".join(f"{a.scheme_name} → {a.verdict.value}" for a in assessments)
    ]
    return {"assessments": assessments, "trace": trace}
